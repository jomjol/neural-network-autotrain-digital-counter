import script_automate.resize

#Allgemeine Einstellungen
exec(open("../configuration.py").read())

if LogFile != None:
    sys.stdout = open(LogFile, 'a') 

script_automate.resize.empty_directory(Output_Resize)
script_automate.resize.image_resize(Input_Raw, Output_Resize, target_size_x, target_size_y)





