import os
import sys
import json
import struct
import platform
import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
import sounddevice as sd
import time

# Python du venv où mlx fonctionne (surchargeable via la variable d'environnement MLX_PYTHON)
MLX_PYTHON = os.environ.get("MLX_PYTHON", os.path.expanduser("~/.crearobot_mlx/bin/python"))
WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlx_worker.py")


def _default_backend():
    """ MLX (Metal) sur Mac Apple Silicon, faster-whisper partout ailleurs.
    Surchargeable via CREAROBOT_STT_BACKEND=mlx|faster_whisper """
    forced = os.environ.get("CREAROBOT_STT_BACKEND")
    if forced:
        return forced
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "faster_whisper"


class MLXBackend:
    """ Whisper via MLX dans un sous-processus (venv séparé, voir mlx_worker.py) """

    def __init__(self, logger):
        logger.info("Démarrage du worker Whisper MLX...")
        # Environnement propre : sans les variables de ROS/conda qui perturbent le venv
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONHOME") and not k.startswith("DYLD_")}
        self.worker = subprocess.Popen(
            [MLX_PYTHON, WORKER_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
        )
        line = self.worker.stdout.readline()
        if not line or not json.loads(line).get("ready"):
            raise RuntimeError("Le worker MLX n'a pas démarré (voir les erreurs ci-dessus)")
        logger.info("--- WHISPER MLX (METAL) PRET ---")

    def transcribe(self, audio):
        """ Renvoie (texte, fin du dernier segment en secondes) """
        self.worker.stdin.write(struct.pack("<I", len(audio)))
        self.worker.stdin.write(audio.tobytes())
        self.worker.stdin.flush()
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("Le worker MLX s'est arrêté")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        segments = result["segments"]
        return result["text"], (segments[-1]["end"] if segments else 0)

    def close(self):
        self.worker.terminate()


class FasterWhisperBackend:
    """ Whisper via faster-whisper : GPU (CUDA) si disponible, sinon CPU """

    def __init__(self, logger):
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel("small", device="cuda", compute_type="float16")
            logger.info("--- WHISPER GPU PRET ---")
        except Exception:
            logger.warning("GPU non disponible — fallback CPU")
            self.model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("--- WHISPER CPU PRET ---")

    def transcribe(self, audio):
        """ Renvoie (texte, fin du dernier segment en secondes) """
        segments, _ = self.model.transcribe(audio, language="fr", vad_filter=True)
        full_text = ""
        last_end  = 0
        for segment in segments:
            full_text += segment.text
            last_end   = segment.end
        return full_text, last_end

    def close(self):
        pass


_BACKENDS = {"mlx": MLXBackend, "faster_whisper": FasterWhisperBackend}


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, '/pc/stt/transcript', 10)
        self.create_subscription(Bool, '/pc/stt/enable', self.enable_callback, _latched)
        self.create_subscription(Bool, '/pc/robot/speaking', self.speaking_callback, 10)

        self.enabled          = False
        self.robot_speaking   = False  # prioritaire sur enabled
        self.audio_buffer     = []
        self._mic_open_logged = False

        name = _default_backend()
        if name not in _BACKENDS:
            raise ValueError(f"CREAROBOT_STT_BACKEND inconnu : {name} (attendu : {', '.join(_BACKENDS)})")
        self.get_logger().info(f"Moteur STT : {name}")
        self.backend = _BACKENDS[name](self.get_logger())

    @property
    def _listening(self):
        return self.enabled and not self.robot_speaking

    def enable_callback(self, msg):
        self.enabled = msg.data
        if not msg.data:
            self.audio_buffer = []

    def speaking_callback(self, msg):
        self.robot_speaking = msg.data
        if msg.data:
            self.audio_buffer = []
            self._mic_open_logged = False
        elif self.enabled and not self._mic_open_logged:
            self._mic_open_logged = True
            self.get_logger().info("Micro ouvert — en écoute")

    # Patterns d'hallucination connus de Whisper sur silence/bruit
    _HALLUCINATIONS = [
        "sous-titres réalisés par",
        "amara.org",
        "sous-titrage st'501",
        "merci d'avoir regardé",
        "abonnez-vous",
        "sous-titres par",
        "transcription by",
        "subtitles by",
        "translated by",
    ]

    def _is_hallucination(self, text: str) -> bool:
        t = text.lower()
        return any(h in t for h in self._HALLUCINATIONS)

    def audio_callback(self, indata, frames, time, status):
        if self._listening:
            self.audio_buffer.append(indata.copy())

    def process_live(self):
        if not self._listening or not self.audio_buffer:
            return

        recording = np.concatenate(self.audio_buffer, axis=0).flatten().astype(np.float32)
        if len(recording) < 16000 * 0.5:
            return

        try:
            t0 = time.time()
            full_text, last_end = self.backend.transcribe(recording)
            full_text = full_text.strip()

            duration_total = len(recording) / 16000
            silence_at_end = duration_total - last_end

            if full_text and silence_at_end > 0.9:
                if self._is_hallucination(full_text):
                    self.get_logger().warning(f"[HALLUCINATION filtrée] {full_text}")
                    self.audio_buffer = []
                    return
                self.get_logger().info(f"[LATENCE] Whisper transcription : {time.time() - t0:.2f} s")
                self.get_logger().info(f"Phrase detectee : {full_text}")
                msg = String()
                msg.data = full_text
                self.pub.publish(msg)
                self.audio_buffer = []

        except Exception as e:
            self.get_logger().error(f"Erreur transcription : {e}")
            self.audio_buffer = []


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()

    try:
        with sd.InputStream(samplerate=16000, channels=1, callback=node.audio_callback):
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                node.process_live()
    finally:
        node.backend.close()
        node.destroy_node()
        rclpy.shutdown()
