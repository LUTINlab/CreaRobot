import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float64MultiArray
import roslibpy
import json
import os

from .task_synchronizer import TaskSynchronizer
from . import config


class GatewayNode(Node):
    def __init__(self):
        super().__init__('gateway_node')

        self.ts = TaskSynchronizer()

        # Connexion au robot (ROS 1)    
        self.get_logger().info('Connexion au robot QT')
        self.ros1_client = roslibpy.Ros(host="192.168.24.118", port=9091)
        self.ros1_client.run()

        # Topic ROS 1 pour le contrôle de la tête
        self.head_pub_ros1 = roslibpy.Topic(self.ros1_client, '/qt_robot/head_position/command', 'std_msgs/Float64MultiArray')

        # Services ROS 1 du robot
        self.talk_srv         = roslibpy.Service(self.ros1_client, '/qt_robot/behavior/talkText',  'qt_robot_interface/behavior_talk_text')
        self.gesture_srv      = roslibpy.Service(self.ros1_client, '/qt_robot/gesture/play',       'qt_robot_interface/gesture_play')
        self.emotion_srv      = roslibpy.Service(self.ros1_client, '/qt_robot/emotion/show',       'qt_robot_interface/emotion_show')
        self.voice_config_srv = roslibpy.Service(self.ros1_client, '/qt_robot/speech/config',      'qt_robot_interface/speech_config')
        self.volume_srv       = roslibpy.Service(self.ros1_client, '/qt_robot/setting/setVolume',  'qt_robot_interface/setting_setVolume')

        self.ros1_client.on_ready(self.setup_robot_voice)

        # Publisher : indique au STT si le robot est en train de parler
        self.speaking_pub = self.create_publisher(Bool, '/pc/robot/speaking', 10)

        # Subscribers ROS 2
        self.create_subscription(String,            '/pc/qtaction',               self.listener_callback, 10)
        self.create_subscription(Float64MultiArray, '/pc/head_position/command',  self.head_callback,     10)

        self.get_logger().info("Gateway Active")

    def setup_robot_voice(self):
        try:
            self.voice_config_srv.call(roslibpy.ServiceRequest({
                'language': config.VOICE_LANG,
                'pitch'   : config.VOICE_PITCH,
                'speed'   : config.VOICE_SPEED
            }))
        except Exception as e:
            self.get_logger().error(f"ERREUR voice_config: {e}")
        try:
            self.volume_srv.call(roslibpy.ServiceRequest({'volume': int(config.VOICE_VOLUME)}))
        except Exception as e:
            self.get_logger().error(f"ERREUR setVolume: {e}")

    def listener_callback(self, msg):
        try:
            data = json.loads(msg.data)
            geste, emotion, texte = data[0], data[1], data[2]

            has_text   = texte   and texte.strip()   != ""
            no_geste   = not geste   or geste.lower()   == "none"
            no_emotion = not emotion or emotion.lower() == "none"

            # Texte pur — envoi direct sans sync
            if has_text and no_geste and no_emotion:
                self.get_logger().info(f'Direct Talk : "{texte}"')
                self._publish_speaking(True)
                self.talk_srv.call(roslibpy.ServiceRequest({'message': texte}))
                self._publish_speaking(False)
                return

            # Animation pure (pas de texte)
            if not has_text:
                tasks = []
                if not no_emotion:
                    tasks.append((0, lambda: self.emotion_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + emotion}))))
                if not no_geste:
                    tasks.append((0, lambda: self.gesture_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 1.0}))))
                if tasks:
                    self.get_logger().info("Animation seule")
                    self.ts.sync(tasks)
                return

            # Action combinée (texte + geste/émotion)
            self.get_logger().info(f'Action Synchro : {geste} | {emotion} | "{texte}"')
            self._publish_speaking(True)
            tasks = [
                (0, lambda: self._talk_and_notify(texte)),
                (0, lambda: self.emotion_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + emotion}))),
                (0, lambda: self.gesture_srv.call(roslibpy.ServiceRequest({'name': 'QT/' + geste, 'speed': 1.0})))
            ]
            self.ts.sync(tasks)

        except Exception:
            if msg.data and msg.data.strip() != "":
                self.talk_srv.call(roslibpy.ServiceRequest({'message': msg.data}))

    def _publish_speaking(self, is_speaking: bool):
        msg = Bool()
        msg.data = is_speaking
        self.speaking_pub.publish(msg)

    def _talk_and_notify(self, texte: str):
        self.talk_srv.call(roslibpy.ServiceRequest({'message': texte}))
        self._publish_speaking(False)

    def head_callback(self, msg):
        try:
            self.head_pub_ros1.publish(roslibpy.Message({'data': list(msg.data)}))
        except Exception:
            self.get_logger().error("Erreur envoi tête ROS 1")


def main(args=None):
    rclpy.init(args=args)
    node = GatewayNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        os._exit(0)
