# Orchestration expérience TCT-DP

> QTrobot V1 et V2 (ROS 1 Kinetic et ROS 1 Noetic) transformé en compagnon social intelligent. L'intelligence, la vision et l'audition sont déportées sur PC (ROS 2) pour une réactivité maximale.

---

## Architecture du système

Pour garantir une interaction fluide, les capteurs sont côté PC et indépendants du robot : micro intégré du PC et caméra USB externe. La Gateway ne sert plus qu'à envoyer les commandes motrices et vocales au robot.

![Architecture du système CreaRobot](docs/figures/architecture_système.png)

| Couche | Node | Rôle |
|---|---|---|
| Logique | `orchestrator_node` | Machine à états — 7 phases TCT-DP |
| Dispatcher | `interaction_node` | Traduit les phases en actions (gestes / émotions / parole) |
| Audition | `stt_node` | Capture micro local + STT local (Whisper : MLX sur Mac Apple Silicon, Faster-Whisper sur PC) |
| Vision | `vision_node` | Flux local via caméra USB externe |
| Vision démo | `vision_node_temp` | Mode Science Infuse : capture d'écran HDMI (via mss) |
| Détection dessin | `drawing_detector_node` | Détecte la fin du dessin par frame differencing + médiane glissante |
| Cognition | `brain_node` | LLM/VLM : analyse texte et image, génère les réponses |
| Projection | `projection_node` | Affiche le visuel du dessin sur le projecteur HDMI (C2) |
| Passerelle | `gateway_node` | Bridge ROS 1 ↔ ROS 2 via WebSocket (rosbridge :9091) |

---

## Connexion et configuration

### QTRobot V1

```bash
# SSH
ssh qtrobot@192.168.100.1 # QTRP
ssh qtrobot@192.168.100.2 # QTPC

# Interface web (autostart des services)
http://192.168.100.1:8080
http://192.168.100.2:8080
```

Workflow de développement, accès direct aux fichiers du robot :

```bash
sftp://qtrobot@192.168.100.1/
```

### QTRobot V2

Le V2 ne dispose pas d'interface web ni de hotspot WiFi propre. La Raspberry Pi (QTRP) est connectée directement à un réseau WiFi et le PC doit être connecté au même réseau. La communication ROS 1 ↔ ROS 2 se fait via ce WiFi partagé.

Adresse IP de QTRP sur le réseau :
```
192.168.24.118
```

| Élément | V1 | V2 |
|---|---|---|
| Interface web | ✓ | ✗ |
| Hotspot robot | ✓ | ✗ |
| Connexion | Hotspot du robot | WiFi commun |


---

## Installation

Le projet tourne sur PC Linux (ROS 2 Humble) ou sur Mac Apple Silicon (ROS 2 Jazzy via RoboStack). Seul le moteur de reconnaissance vocale diffère : `stt_node` choisit automatiquement MLX (GPU Metal) sur Mac Apple Silicon et Faster-Whisper ailleurs. Pour forcer un moteur : `export CREAROBOT_STT_BACKEND=mlx` ou `faster_whisper`.

### Sur PC (Linux) - ROS 2 Humble / Python 3.10

#### Dépendances

```bash
pip install roslibpy openai opencv-python "numpy==1.26.4" faster-whisper sounddevice mss nvidia-cublas-cu12 nvidia-cudnn-cu12
sudo apt install ros-humble-cv-bridge
```

#### Whisper sur GPU (recommandé)

Le nœud STT tente de charger Whisper sur GPU (CUDA) au démarrage, avec fallback automatique sur CPU si indisponible. Vérifier que les drivers NVIDIA sont installés (`nvidia-smi`), puis ajouter dans `~/.bashrc` :

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

#### Configuration du Micro

Pour que le stt_node fonctionne de manière optimisée, le microphone du PC doit être défini comme périphérique par défaut.

Listez vos sources : `pactl list short sources`

```bash
pactl set-default-source "NOM_DU_PERIPHERIQUE"
```

### Sur Mac Apple Silicon - ROS 2 Jazzy (RoboStack) / Python 3.11

#### Environnement ROS 2 (conda)

```bash
conda create -n ros_env -c conda-forge -c robostack-jazzy python=3.11 ros-jazzy-desktop ros-jazzy-cv-bridge colcon-common-extensions
conda activate ros_env
pip install roslibpy openai opencv-python "numpy==1.26.4" sounddevice mss
```

#### Whisper MLX (venv séparé)

MLX ne fonctionne pas dans l'environnement conda : `stt_node` lance Whisper dans un sous-processus (`mlx_worker.py`) avec le Python d'un venv dédié, par défaut `~/.crearobot_mlx` (surchargeable via `MLX_PYTHON`).

```bash
/usr/bin/python3 -m venv ~/.crearobot_mlx
~/.crearobot_mlx/bin/pip install mlx-whisper
```

Le modèle `whisper-small-mlx` est téléchargé au premier lancement. Le micro utilisé est celui défini par défaut dans les Réglages Système (Son → Entrée).

#### Condition C2

Pour charger la feuille TCT-DP comme modèle de génération : `brew install poppler` (fournit `pdftoppm`).

---

## Mise en route

### 1. Côté robot - ROS 1 (Si QT V1)

Activer dans l'autostart ros_bridge et motor :
- *start_qt_rosbridge.sh*
- *start_qt_motor.sh*

### 2. Côté PC - ROS 2

```bash
colcon build
source install/setup.bash   # sur Mac : source install/setup.zsh
```

Sur Mac, `./run.sh` active `ros_env` et lance l'infrastructure et l'orchestrateur en une commande.

#### Méthode principale (recommandée)

Terminal 1 - Infrastructure :

```bash
ros2 launch crearobot_brain launch.py
```

Terminal 2 - Pilote de l'expérience :

```bash
ros2 run crearobot_brain orchestrator
```

#### Méthode alternative (noeuds séparés)

```bash
ros2 run crearobot_brain interaction
ros2 run crearobot_brain orchestrator
ros2 run crearobot_brain brain
ros2 run crearobot_brain gateway
ros2 run crearobot_brain projection
ros2 run crearobot_brain stt
ros2 run crearobot_brain vision            # caméra USB
ros2 run crearobot_brain vision_temp       # capture écran HDMI
ros2 run crearobot_brain drawing_detector 
```

---

## Structure de l'expérience

![Session du système CreaRobot](docs/figures/session_crearobot.png)

### Détail des phases

| Phase | Commande | Description |
|---|---|---|
| Introduction | `START_INTRO` | Accueil du participant, présentation du robot |
| Ice Breaking | `START_ICE_BREAKING` | 3 échanges amicaux pilotés par le LLM |
| Consignes | `START_TASK_INTRO` | Explication de l'activité TCT-DP |
| Dessin (×3) | `START_DRAWING_X` | 90s de dessin, angle de tête configurable (`HEAD_PITCH_DRAWING`) |
| Feedback (×2) | `START_FEEDBACK_X` | C0 : 1 phrase générique positive / C1 : VLM + 2 échanges contextualisés (LLM) / C2 : VLM + génération d'image |
| Titre | `START_TITLE` | Analyse VLM finale, génération d'un titre pour le dessin |
| Conclusion | `START_ENDING` | Au revoir et fin de l'expérience |

---

## Fonctionnalités clés

### Synchronisation parfaite (lipsync)

La classe `TaskSynchronizer` (basée sur `asyncio`) déclenche simultanément :
- le geste (ROS Service)
- l'expression faciale (ROS Service)
- la parole (TTS)

Un correctif dynamique soustrait le temps de chargement de l'émotion à la durée d'animation de la bouche.

### Détection automatique de fin de dessin

Le `drawing_detector_node` surveille la caméra en continu pendant la phase de dessin et détecte quand le participant arrête de dessiner, via **frame differencing avec médiane glissante** :

- À chaque tick (toutes les `FRAME_DIFF_INTERVAL` secondes), deux frames consécutives sont comparées (`cv2.absdiff`)
- Le nombre de pixels différents est ajouté à une fenêtre glissante de `FRAME_DIFF_WINDOW_SIZE` ticks
- Si la **médiane** de la fenêtre est sous `FRAME_DIFF_PIXEL_THRESHOLD`, la période est considérée comme calme
- Après `FRAME_DIFF_INACTIVITY_DURATION` secondes consécutives de calme, un `Bool(True)` est publié sur `/pc/vision/drawing_stopped`
- L'`orchestrator_node` reçoit ce signal et déclenche immédiatement la transition de phase (comme si le timer avait expiré)

La médiane est préférée à une simple valeur instantanée pour absorber les pics ponctuels (toussotement, micro-mouvement). Le timer `DRAW_DURATION` reste le mécanisme principal ; le détecteur permet de terminer plus tôt si le participant retire sa main du cadre.


### Vision & projection

Le `vision_node` récupère un flux local via caméra USB (index configurable via `CAMERA_INDEX`). Le `projection_node` utilise OpenCV pour mapper une fenêtre plein écran sur la sortie HDMI du projecteur (condition C2).

### Détection d'adressage en temps réel

Pendant la phase de dessin, chaque transcription STT est analysée par un LLM (`brain_node`) pour déterminer si le participant s'adresse au robot. Le résultat est publié sur `/pc/brain/addressing_detected`.

- Si **TRUE** : réponse immédiate à l'aveugle (courte phrase adaptée au contenu, avec instruction de reculer pour la caméra), puis transition vers FEEDBACK avec `triggered_by="addressing"`
- La phrase du participant est mémorisée dans `_addressing_user_input` et injectée dans le prompt VLM après la prise de photo, pour une réponse contextualisée
- Pattern "latest wins" : si plusieurs transcriptions arrivent pendant qu'un check LLM est en cours, seul le dernier est traité
- Guard de phase : si la phase change entre le début du check et sa conclusion, la transition est ignorée

### Audition déportée - STT

Le nœud `stt_node` embarque Faster-Whisper en local (modèle `small` sur GPU, `small` sur CPU en fallback). Il est piloté par le topic `/pc/stt/enable` (Bool, QoS `TRANSIENT_LOCAL` pour éviter les race conditions au démarrage).

Un filtre d'hallucinations élimine les artefacts connus de Whisper qui apparaissent comme de la pub (ex : "Sous-titres réalisés par la communauté d'Amara.org").

#### Verrouillage micro (Half-Duplex)

Pour éviter que QT ne s'écoute parler :
- Dès que QT commence une phrase, `is_busy` passe à `True`
- Le micro est coupé (`set_stt(False)`)
- Il n'est réactivé qu'une fois la durée théorique de la phrase écoulée

### Intelligence artificielle : LLM/VLM

- **Analyse multimodale** : GPT-4o-mini croise données visuelles (dessin) et transcriptions STT
- **Mémoire cumulative** : description textuelle du dessin enrichie à chaque analyse, sans retraiter les pixels

### Mouvements de tête dynamiques

- **Phase Dessin** : angle de tête configurable via `HEAD_PITCH_DRAWING` dans `config.py` (par défaut 0, tête droite)
- **Phase Feedback / Ice Breaking** : tête droite (HeadPitch = 0) - contact visuel avec le participant

### Journalisation automatique des sessions

À chaque lancement, le système crée automatiquement un dossier de session horodaté dans `crearobot_sessions/` à la racine du repo (ignoré par git) :

```
crearobot_sessions/
  2026-06-29/
    14-32-05/
      conversation.log   ← transcriptions PARTICIPANT + textes ROBOT + transitions de phase
      dessins/           ← photos du dessin réel (vision_node)
        01_feedback_tour1_14-33-12.jpg
        02_feedback_tour2_14-41-07.jpg
        03_title_tour3_14-52-44.jpg
      dessins_generes/   ← images IA projetées, condition C2 (projection_node)
        01_genere_14-33-55.jpg
        02_genere_14-42-10.jpg
```

**`conversation.log`** — enregistré par `brain_node`, horodaté au format `HH:MM:SS` :
```
=== Session CreaRobot — 2026-06-29 14-32-05 — Condition C1 ===

[14:32:06] PHASE        START_INTRO  Tour 1
[14:32:20] PHASE        START_ICE_BREAKING  Tour 1
[14:33:00] PHASE        START_DRAWING  Tour 1
[14:33:10] PARTICIPANT : j'ai fait un bonhomme avec un chapeau
[14:33:12] PHASE        START_FEEDBACK  Tour 1  ← ADRESSAGE
[14:33:14] ROBOT       : Oh intéressant ! Tu as pensé à lui donner une expression ?
[14:35:00] PHASE        START_DRAWING  Tour 2
[14:36:30] PHASE        START_FEEDBACK  Tour 2  ← TIMEOUT
[14:38:00] PHASE        START_DRAWING  Tour 3
[14:39:10] PHASE        START_FEEDBACK  Tour 3  ← FRAME DIFFERENCING
```

Les lignes `PHASE` tracent chaque transition d'état. Pour les sorties de phase `DRAWING`, la cause est indiquée : `ADRESSAGE` (participant s'est adressé au robot), `TIMEOUT` (durée max atteinte), `FRAME DIFFERENCING` (inactivité détectée), ou `TERMINAL (fini/skip/…)` (commande manuelle).

**`dessins/`** — photos capturées par `vision_node` à chaque déclenchement caméra, nommées `{index}_{phase}_tour{N}_{heure}.jpg`.

**`dessins_generes/`** — images générées par l'IA et projetées sur la feuille (condition C2 uniquement), sauvegardées par `projection_node` à chaque réception sur `/pc/projector/image`.

**Synchronisation des nœuds** — `brain_node` publie le chemin de session sur `/pc/session_dir` avec une QoS `TRANSIENT_LOCAL` (message latché). `vision_node` et `projection_node` s'y abonnent et créent leurs sous-dossiers dès réception, quel que soit l'ordre de démarrage.

---

## Tests

| Fichier | Description |
|---|---|
| `client_bridge.py` | Test de la gateway, communication robot ↔ PC |
| `test_gpt.py` | Validation du format JSON et du respect des listes de gestes/émotions |
| `bridge_client_cam.py` | Affichage du flux vidéo du robot en temps réel |
| `test_sound_treating.py` | Test du pipeline audio : capture micro + transcription Faster-Whisper |


---

## Stats

Les analyses portent sur les scores TCT-DP (Urban, 1991) collectés lors de la passation pilote (Science Infuse). Deux conditions sont comparées : **C0** (feedback neutre) et **C1** (feedback LLM/VLM). Les Adultes sont exclus de toutes les analyses (N=3, groupe déséquilibré).

### Normalisation

Tous les critères TCT-DP sont normalisés sur leur score maximum avant analyse :
`Cn, Cm, Ne, Cl, Cth, Pe /6` · `Bfd, Bfi, Hu /3` · `Uc_b, Uc_c, Uc_d /2`

Le score total normalisé est la somme des 12 critères normalisés → **max = 12**.

### Données

`scores_tctdp.csv` - colonnes : `Participant`, `Condition` (C0/C1), `Tranche_age`, 12 critères TCT-DP, `Total`.

`score_todd.csv` - note de créativité subjective attribuée par plusieurs experts.

| Tranche d'âge | C0 | C1 | Inclus |
|---|---|---|---|
| Enfant | 20 | 17 | ✓ |
| Jeune enfant | 5 | 5 | ✓ |
| Adolescent | 1 | 1 | ✓ |
| Adulte | 0 | 3 | ✗ |

### Scripts (`stats/scripts/`)

| Script | Description |
|---|---|
| `t_test.py` | T-test de Welch C0 vs C1 sur le score total brut, retrait outliers |
| `criteres_scoring.py` | Cohen's d et moyennes normalisées par critère TCT-DP (C0 vs C1) |
| `analyse_criteres.py` | T-tests bilatéraux par critère normalisé + Cohen's d |
| `score_expert.py` | Analyse du score expert, note de créativité subjective |
| `pca_clustering.py` | ACP sur les 12 critères normalisés ; scree plot, espace PCA, contributions (cos²) |
| `clustering_criteres.py` | Clustering K-means (k=5) sur les critères ; profils créatifs A/B/C/D/E |
| `pca_sensitivity.py` | Analyse de sensibilité PCA : contribution des critères aux composantes (cos²) |

```bash
cd stats/scripts/
python3 nom_script.py
```

### Figures (`stats/figures/`)

| Fichier | Généré par |
|---|---|
| `ttest.png` | `t_test.py` |
| `criteres_scoring.png` | `criteres_scoring.py` |
| `criteres_cohens_d.png` | `analyse_criteres.py` |
| `score_expert.png` | `score_expert.py` |
| `pca_clustering.png` | `pca_clustering.py` |
| `clustering_criteres.png` | `clustering_criteres.py` |
| `clustering_validation.png` | `clustering_criteres.py` |
| `clustering_stability.png` | `clustering_criteres.py` |
| `pca_sensitivity.png` | `pca_sensitivity.py` |

---

## Sources

- [Documentation officielle QTrobot](https://docs.luxai.com/docs/intro_code)
- [Wiki ROS QTrobot](https://wiki.ros.org/Robots/qtrobot)

---

*Développé par Marco G.*
