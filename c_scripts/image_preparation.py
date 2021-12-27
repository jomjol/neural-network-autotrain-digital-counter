import c_scripts.resize as resize

#Allgemeine Einstellungen
exec(open("configuration.py").read())

if LogFile != None:
    sys.stdout = open(LogFile, 'a') 

resize.empty_directory(Output_Resize)
resize.image_resize(Input_Raw, Output_Resize, target_size_x, target_size_y)





