from launch import LaunchDescription
from launch_ros.actions import Node
from crearobot_brain import config

def generate_launch_description():
    
    # Gateway (Pont vers le corps du robot)
    gateway = Node(
        package='crearobot_brain',
        executable='gateway',
        name='gateway_node',
        output='screen'
    )

    # Projection (Sortie HDMI Projecteur)
    projection = Node(
        package='crearobot_brain',
        executable='projection',
        name='projection_node',
        output='screen'
    )

    # Vision (Capture Caméra)
    vision = Node(
        package='crearobot_brain',
        executable='vision',
        name='vision_node',
        output='screen'
    )

    
    # Vision (Capture écran HDMI) Semaine science infuse
    # Pour l'utiliser, décommenter et remplacer `vision` par `vision_temp` dans nodes_to_run
    # vision_temp = Node(
    #     package='crearobot_brain',
    #     executable='vision_temp',
    #     name='vision_node_temp',
    #     output='screen'
    # )
    

    # STT (Oreilles - Micro USB)
    stt = Node(
        package='crearobot_brain',
        executable='stt',
        name='stt_node',
        output='screen'
    )

    # Interaction (Logique de dialogue et Feedback C1/C2)
    interaction = Node(
        package='crearobot_brain',
        executable='interaction',
        name='interaction_node',
        output='screen'
    )

    # Brain (Dialogue sur le dessin avec le participant)
    brain = Node(
        package='crearobot_brain',
        executable='brain',
        name='brain_node',
        output='screen'
    )

    # Drawing detection (Detecte si le participant a fini de dessiner)
    drawing_detector = Node(
        package='crearobot_brain',
        executable='drawing_detector',
        name='drawing_detector_node',
        output='screen'
    ) 
    
    
    nodes_to_run = [gateway, stt, interaction, brain, vision, drawing_detector]

    if config.CONDITION == "C2":
        nodes_to_run.append(projection)

    return LaunchDescription(nodes_to_run)