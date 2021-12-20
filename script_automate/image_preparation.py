import script_automate.resize

#Allgemeine Einstellungen
exec(open("configuration.py").read())

script_automate.resize.empty_directory(Output_Resize)
script_automate.resize.image_resize(Input_Raw, Output_Resize, target_size_x, target_size_y)





