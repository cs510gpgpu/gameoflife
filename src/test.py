import os
import platform

platformWinOrNix = platform.system()


if platformWinOrNix ==  'Windows':
    print("You are running Windows")
else:
    print("You are running Linux")
