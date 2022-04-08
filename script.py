import sys
import os

exec(open("configuration.py").read())

original_stdout = sys.stdout
if LogFile:
    if os.path.exists(LogFile):
        os.remove(LogFile)

#if not ReportOnly:
#    exec(open("c_scripts/image_preparation.py").read())
    
#for x in configurations:
#    arg1 = x
#    arg2 = x
#    if not ReportOnly:
#        print("Das ist in Script - Teil 1: " + str(arg1))
#        a_script = open("c_scripts//doTraining.py").read()
#        sys.argv = ["c_scripts//doTraining.py", arg1]
#        exec(a_script)
#    print("Das ist in Script - Teil 2: " + str(arg2))
#    b_script = open("c_scripts//doReport.py").read()
#    sys.argv = ["c_scripts//doReport.py", arg2]
#    exec(b_script)

# alternative Model-Training 
exec(open("d_scripts_effnet/train.py").read())
    
sys.stdout = original_stdout