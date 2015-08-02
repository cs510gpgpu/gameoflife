import os
import platform
import subprocess

platformWinOrNix = platform.system()


if platformWinOrNix ==  'Windows':
    print("You are running Windows")
    # basic output
    run1 = subprocess.Popen(["nvprof", "--log-file", "test1.txt", "--csv", "--timeout", "30", "..\\Release\\gameoflife.exe", "1024", "768", "OpenGL"], stdout=subprocess.PIPE)
    run1.stdout.readlines()
    # visual profiler output                                                                                                    #these can be put in a loop and run at many variations
    run2 = subprocess.Popen(["nvprof", "--analysis-metrics", "-o", "run1.nvprof", "--timeout", "30", "..\\Release\\gameoflife.exe", "1024", "768", "OpenGL"], stdout=subprocess.PIPE)
    run3.stdout.readlines()
else:
    print("You are running Linux")
