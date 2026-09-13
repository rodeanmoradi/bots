"""Action Observation Therapy loop for BracketBot.

    demonstrate (sim robot)  ->  observe (webcam)  ->  score  ->  adapt  ->  repeat

Robot and child are compared in the same representation: where the hand goes
relative to the shoulder, in units of arm length (see `motion.Motion`).
Nothing in this package imports ROS; the robot is reached through
`robot_link` (a stub, or a socket to `play_demo.py` on the sim machine).
"""
