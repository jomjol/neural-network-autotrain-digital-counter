import sys
exec(open("configuration.py").read())
exec(open("script_automate/image_preparation.py").read())
for x in configurations:
    arg1 = x
    arg2 = x
    print("Das ist in Script - Teil 1: " + str(arg1))
    a_script = open("script_automate/doTraining.py").read()
    sys.argv = ["script_automate/doTraining.py", arg1]
    exec(a_script)
    print("Das ist in Script - Teil 2: " + str(arg2))
    b_script = open("script_automate/doReport.py").read()
    sys.argv = ["script_automate/doReport.py", arg2]
    exec(b_script)