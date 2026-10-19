import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy
import openai
import json
import os
import base64
import io
import subprocess
import tempfile
import glob
import time
import datetime
import numpy as np
import cv2
import threading
from sensor_msgs.msg import Image

from . import config


class BrainNode(Node):

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('brain_node')
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY, timeout=90.0)

        self.visual_memory = ""
        self.chat_history  = []
        self.current_phase = ""
        self.current_tour  = 0
        self._c1_ready      = False  # True seulement après la 1ère analyse photo du feedback
        self._c1_robot_turn = 0     # Numéro de prise de parole du robot dans le feedback courant
        self._addressing_triggered  = False  # True si adressage déjà détecté ce tour
        self._addressing_checking   = False  # True si un check LLM est en cours
        self._addressing_pending    = None   # dernier transcript reçu pendant un check en cours
        self._addressing_user_input = None   # phrase du participant qui a déclenché l'adressage

        self.create_subscription(String, '/pc/vision/image_raw_b64', self.process_vision,  10)
        self.create_subscription(String, '/pc/stt/transcript',        self.handle_dialogue, 10)
        self.create_subscription(String, '/pc/phase_control',         self.phase_callback,  10)

        self.tts_pub         = self.create_publisher(String, '/pc/vision/feedback',           10)
        self.image_pub       = self.create_publisher(Image,  '/pc/projector/image',           10)
        self.qtaction_pub    = self.create_publisher(String, '/pc/qtaction',                  10)
        self.addressing_pub  = self.create_publisher(Bool,   '/pc/brain/addressing_detected', 10)

        if config.CONDITION == "C2":
            self.tctdp_template_b64 = self._load_tctdp_template()
            self.tctdp_template_png_buf = self._precompute_template_png()
        else:
            self.tctdp_template_b64 = None
            self.tctdp_template_png_buf = None

        # ── Logger de session ─────────────────────────────────────────────────
        now = datetime.datetime.now()
        day_str  = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        self.session_dir = os.path.join(config.SESSIONS_DIR, day_str, time_str)

        os.makedirs(self.session_dir, exist_ok=True)
        log_path = os.path.join(self.session_dir, "conversation.log")
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._log_file.write(
            f"=== Session CreaRobot — {day_str} {time_str} — Condition {config.CONDITION} ===\n\n"
        )
        self._log_file.flush()

        # Publie le chemin de session (latched) pour que vision_node utilise le même dossier
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._session_dir_pub = self.create_publisher(String, '/pc/session_dir', _latched_qos)
        _sd_msg = String()
        _sd_msg.data = self.session_dir
        self._session_dir_pub.publish(_sd_msg)

        # Abonnement au transcript participant (pour le log)
        self.create_subscription(String, '/pc/stt/transcript',   self._log_participant, 10)
        # Abonnement aux actions robot (pour le log du texte robot)
        self.create_subscription(String, '/pc/qtaction',         self._log_robot,       10)

        self.warmup_llm()
        self.get_logger().info(f"Brain Node prêt — session : {self.session_dir}")

    # ── Gestion des phases ────────────────────────────────────────────────────

    def phase_callback(self, msg):
        data = json.loads(msg.data)
        self.current_phase = data["phase"]
        self.current_tour  = data["tour"]
        triggered_by = data.get("triggered_by", "")
        self._log_phase(self.current_phase, self.current_tour, triggered_by)

        if "START_FEEDBACK" in self.current_phase:
            self._c1_ready = False
            self._c1_robot_turn = 0

        if "START_DRAWING" in self.current_phase:
            self._addressing_triggered  = False
            self._addressing_checking   = False
            self._addressing_pending    = None
            self._addressing_user_input = None

        if "START_ICE_BREAKING" in self.current_phase:
            if not self.chat_history:
                self.chat_history.append({
                    "role": "assistant",
                    "content": config.ICE_BREAKING_QUESTION
                })

        if "START_TASK_INTRO" in self.current_phase:
            if self.chat_history:
                self.get_logger().info("Réinitialisation mémoire : Ice Breaking effacé.")
                self.chat_history = []

    # ── Analyse visuelle ──────────────────────────────────────────────────────

    def process_vision(self, msg):
        self.get_logger().info("Analyse du dessin en cours...")
        try:
            is_title_phase = "START_TITLE" in self.current_phase

            if config.CONDITION == "C2" and not is_title_phase:
                c2_messages = [{"role": "user", "content": [
                    {"type": "text", "text": config.PROMPT_C2_ANALYSIS},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg.data}"}}
                ]}]
                t0 = time.time()
                c2_resp = self.client.chat.completions.create(
                    model="gpt-4o-mini", messages=c2_messages, max_tokens=100
                )
                self.get_logger().info(f"[LATENCE] GPT vision C2 : {time.time() - t0:.2f} s")
                self.generate_c2_feedback(msg.data, c2_resp.choices[0].message.content)
                return

            if is_title_phase:
                prompt = config.PROMPT_TITLE
            elif not self.visual_memory:
                prompt = (
                    "Décris ce que tu vois sur le dessin de maniere a pouvoir te rappeler "
                    "du dessin plus tard sans le voir (entre autre, précise les formes et "
                    "les objets de ce dessin)."
                )
            else:
                prompt = (
                    f"Precedemment, le dessin etait : {self.visual_memory}. "
                    "Dis moi ce qui a change ou ce qui est nouveau ou les modifications "
                    "apportées par rapport à la description précédente. Sois très précis sur les ajouts."
                )

            messages = list(self.chat_history)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg.data}"}}
                ]
            })

            t0 = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300
            )
            self.get_logger().info(f"[LATENCE] GPT vision : {time.time() - t0:.2f} s")
            answer = response.choices[0].message.content

            if is_title_phase:
                self.send_to_robot(answer)
                return

            if not self.visual_memory:
                self.visual_memory = answer
            else:
                self.visual_memory += f"\n[Ajouts récents] : {answer}"

            if config.CONDITION == "C1":
                self.generate_c1_feedback()

        except Exception as e:
            self.get_logger().error(f"Erreur Vision : {e}")

    # ── Feedback C1 (dialogue verbal) ─────────────────────────────────────────

    def generate_c1_feedback(self):
        try:
            self._c1_robot_turn += 1
            turn_instruction = self._c1_turn_instruction()
            messages = [{"role": "system", "content": config.PROMPT_C1}]
            messages.extend(self.chat_history)
            messages.append({"role": "system", "content": f"Dessin actuel : {self.visual_memory}"})
            if self._addressing_user_input:
                messages.append({"role": "system", "content": (
                    f"Le participant vient de s'adresser à toi en disant : « {self._addressing_user_input} ». "
                    "Tiens-en compte dans ta réponse (réponds à sa remarque ou question si pertinent)."
                )})
                self._addressing_user_input = None
            messages.append({"role": "system", "content": turn_instruction})

            t0 = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100
            )
            self.get_logger().info(f"[LATENCE] GPT C1 feedback : {time.time() - t0:.2f} s")
            text = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": text})
            self._c1_ready = True  # à partir d'ici, handle_dialogue peut répondre
            self.send_to_robot(text)

        except Exception as e:
            self.get_logger().error(f"Erreur C1 : {e}")

    # ── Feedback C2 (génération d'image) ───────────────────────────────

    def generate_c2_feedback(self, base64_image, description=""):
        try:
            self.get_logger().info("Génération image C2...")

            img_raw = base64.b64decode(base64_image)
            img_arr = np.frombuffer(img_raw, dtype=np.uint8)
            img_cv  = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            _, img_png_buf = cv2.imencode('.png', img_cv)

            t0 = time.time()
            edit_prompt = config.PROMPT_C2_EDIT.format(description=description)
            if self.tctdp_template_png_buf is not None:
                images_for_edit = [
                    ("template.png", io.BytesIO(self.tctdp_template_png_buf.tobytes()), "image/png"),
                    ("drawing.png", io.BytesIO(img_png_buf.tobytes()), "image/png"),
                ]
            else:
                images_for_edit = ("drawing.png", io.BytesIO(img_png_buf.tobytes()), "image/png")

            # ── gpt-image-1.5 ─────────────────────────────────
            # (formes, traits) au lieu de laisser le modele les redessiner/deformer
            image_response = self.client.images.edit(
                model="gpt-image-1.5",
                image=images_for_edit,
                prompt=edit_prompt,
                size="auto",
                quality="low",
            )
            self.get_logger().info(f"[LATENCE] gpt-image-1.5 génération : {time.time() - t0:.2f} s")
            img_bytes   = base64.b64decode(image_response.data[0].b64_json)
            img_out_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_decoded = cv2.imdecode(img_out_arr, cv2.IMREAD_COLOR)

            img_proj = cv2.resize(img_decoded, (1920, 1080))

            # ── Publier vers le projecteur ────────────────────────────────────
            ros_img              = Image()
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.height, ros_img.width = img_proj.shape[:2]
            ros_img.encoding     = 'bgr8'
            ros_img.step         = ros_img.width * 3
            ros_img.data         = img_proj.tobytes()
            self.image_pub.publish(ros_img)
            self.get_logger().info("Image generee et envoyee au projecteur")

            if self._addressing_user_input:
                text = "J'ai entendu ce que tu m'as dit. Voilà ce que j'ai essayé à partir de ton dessin !"
                self._addressing_user_input = None
            else:
                text = "Voilà, j'ai essayé quelque chose à partir de ton dessin !"
            self.chat_history.append({"role": "assistant", "content": text})
            self.send_to_robot(text)

        except Exception as e:
            self.get_logger().error(f"Erreur C2 : {e}")


    def handle_dialogue(self, msg):
        is_feedback     = "START_FEEDBACK"     in self.current_phase
        is_ice_breaking = "START_ICE_BREAKING" in self.current_phase
        is_drawing      = "START_DRAWING"       in self.current_phase

        if is_drawing and not self._addressing_triggered:
            if self._addressing_checking:
                # Un check est déjà en cours — on mémorise ce transcript, il sera traité ensuite
                self._addressing_pending = msg.data
            else:
                self._addressing_checking = True
                self._addressing_pending  = None
                threading.Thread(
                    target=self._check_addressing, args=(msg.data,), daemon=True
                ).start()
            return

        if not is_feedback and not is_ice_breaking:
            return
        if is_feedback and (not self.visual_memory or not self._c1_ready):
            return

        user_input = msg.data

        if is_ice_breaking:
            messages = [{"role": "system", "content": config.PROMPT_ICE_BREAKING}]
        else:
            self._c1_robot_turn += 1
            turn_instruction = self._c1_turn_instruction()
            messages = [{"role": "system", "content": config.PROMPT_C1}]
            messages.append({"role": "system", "content": f"Memoire visuelle : {self.visual_memory}"})
            messages.append({"role": "system", "content": turn_instruction})

        messages.extend(self.chat_history[-(config.HISTORY_LIMIT * 2):])
        messages.append({"role": "user", "content": user_input})

        try:
            t0 = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=80
            )
            self.get_logger().info(f"[LATENCE] GPT dialogue : {time.time() - t0:.2f} s")
            answer = response.choices[0].message.content
            self.chat_history.append({"role": "user",      "content": user_input})
            self.chat_history.append({"role": "assistant", "content": answer})
            self.send_to_robot(answer)

        except Exception as e:
            self.get_logger().error(f"Erreur Dialogue ({self.current_phase}) : {e}")

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _load_tctdp_template(self):
        filename = f"test_sheet_{config.TEST_SHEET_VERSION}.pdf"
        here = os.path.dirname(os.path.abspath(__file__))
        candidats = [
            os.path.join(here, '../../../', filename),
            os.path.join(here, '../../../../../../', filename),
        ]
        pdf_path = next((c for c in candidats if os.path.exists(c)), candidats[0])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # 75 DPI suffit pour l'analyse gpt-4o-mini (positions des éléments)
                subprocess.run(
                    ['pdftoppm', '-r', '75', '-l', '1', '-jpeg', pdf_path, f'{tmpdir}/page'],
                    check=True, capture_output=True
                )
                pages = sorted(glob.glob(f'{tmpdir}/page*.jpg'))
                if not pages:
                    raise FileNotFoundError("Aucune page extraite du PDF")
                with open(pages[0], 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
            self.get_logger().info(f"Template TCT-DP charge ({filename})")
            return b64
        except Exception as e:
            self.get_logger().warning(f"Template TCT-DP non charge : {e}")
            return None

    def _precompute_template_png(self):
        """Pré-calcule le template en PNG 1024x1024 une seule fois au démarrage."""
        if not self.tctdp_template_b64:
            return None
        try:
            tpl_raw = base64.b64decode(self.tctdp_template_b64)
            tpl_arr = np.frombuffer(tpl_raw, dtype=np.uint8)
            tpl_cv  = cv2.imdecode(tpl_arr, cv2.IMREAD_COLOR)
            _, buf = cv2.imencode('.png', tpl_cv)
            self.get_logger().info(f"Template TCT-DP pré-calculé en PNG {tpl_cv.shape[1]}x{tpl_cv.shape[0]}")
            return buf
        except Exception as e:
            self.get_logger().warning(f"Pré-calcul template PNG échoué : {e}")
            return None

    # ── Logger de conversation ────────────────────────────────────────────────

    def _ts(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _log_phase(self, phase, tour, triggered_by=""):
        _TRIGGERS = {
            "addressing":      "ADRESSAGE",
            "frame_diff":      "FRAME DIFFERENCING",
            "timeout":         "TIMEOUT",
            "terminal:fini":   "TERMINAL (fini)",
            "terminal:skip":   "TERMINAL (skip)",
            "terminal:terminate": "TERMINAL (terminate)",
            "terminal:title":  "TERMINAL (title)",
        }
        reason = f" ← {_TRIGGERS.get(triggered_by, triggered_by)}" if triggered_by else ""
        line = f"[{self._ts()}]\tPHASE       : {phase}  Tour {tour}{reason}\n"
        self._log_file.write(line)
        self._log_file.flush()

    def _log_participant(self, msg):
        line = f"[{self._ts()}]\tPARTICIPANT : {msg.data}\n"
        self._log_file.write(line)
        self._log_file.flush()

    def _log_robot(self, msg):
        try:
            data = json.loads(msg.data)
            text = data[2] if len(data) > 2 else ""
        except Exception:
            text = msg.data
        if text and text.strip():
            line = f"[{self._ts()}]\tROBOT       : {text}\n"
            self._log_file.write(line)
            self._log_file.flush()

    def _c1_turn_instruction(self):
        if self._c1_robot_turn >= config.MAX_EXCHANGES:
            return "C'est ta DERNIÈRE prise de parole. Conclus en une phrase chaleureuse. INTERDICTION de poser une question."
        return "Tu DOIS terminer ta réponse par une question ouverte et courte."

    def warmup_llm(self):
        try:
            self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
        except Exception as e:
            self.get_logger().error(f"Echec du warmup gpt-4o-mini : {e}")

        if config.CONDITION == "C2":
            import threading
            threading.Thread(target=self._warmup_image, daemon=True).start()

    def _warmup_image(self):
        try:
            t0 = time.time()
            white = np.full((1024, 1024, 3), 255, dtype=np.uint8)
            _, white_buf = cv2.imencode('.png', white)
            images = [("drawing.png", io.BytesIO(white_buf.tobytes()), "image/png")]
            if self.tctdp_template_png_buf is not None:
                images.insert(0, ("template.png", io.BytesIO(self.tctdp_template_png_buf.tobytes()), "image/png"))
            self.client.images.edit(
                model="gpt-image-1.5",
                image=images,
                prompt=".",
                size="1024x1024",
                quality="low",
            )
            self.get_logger().info(f"[WARMUP] gpt-image-1.5 OK ({time.time() - t0:.1f}s)")
        except Exception as e:
            self.get_logger().warning(f"[WARMUP] gpt-image-1.5 échoué : {e}")

    def _check_addressing(self, user_input):
        try:
            messages = [
                {"role": "system", "content": config.PROMPT_ADDRESSING_DETECTION},
                {"role": "user",   "content": user_input}
            ]
            t0 = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=60,
                response_format={"type": "json_object"}
            )
            self.get_logger().info(f"[LATENCE] GPT adressage : {time.time() - t0:.2f} s")
            data    = json.loads(response.choices[0].message.content)
            adresse = data.get("adresse_robot", False)
            reponse = data.get("reponse", "")
            self.get_logger().info(f"[ADRESSAGE] '{user_input}' → adresse={adresse}")

            if adresse:
                if "START_DRAWING" not in self.current_phase:
                    self.get_logger().info("[ADRESSAGE] Phase changée entre-temps — ignoré")
                    self._addressing_pending = None
                    return
                self._addressing_triggered  = True
                self._addressing_user_input = user_input
                if reponse:
                    action = json.dumps(["neutral", "neutral", reponse])
                    qt_msg = String()
                    qt_msg.data = action
                    self.qtaction_pub.publish(qt_msg)
                bool_msg = Bool()
                bool_msg.data = True
                self.addressing_pub.publish(bool_msg)
        except Exception as e:
            self.get_logger().error(f"Erreur détection adressage : {e}")
        finally:
            pending = self._addressing_pending
            self._addressing_pending  = None
            self._addressing_checking = False
            # S'il y avait un transcript en attente et qu'on n'a pas encore détecté d'adressage
            if pending and not self._addressing_triggered:
                self._addressing_checking = True
                threading.Thread(
                    target=self._check_addressing, args=(pending,), daemon=True
                ).start()

    def send_to_robot(self, text):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node._log_file.close()
        except Exception:
            pass
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        os._exit(0)
