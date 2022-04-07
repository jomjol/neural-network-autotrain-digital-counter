from models.distiller import Distiller
from models.cnn32 import CNN32C3C3C5_BN_DA
from models.effnet import Effnet
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence

def create_distiller_models(input_shape, nb_classes):
    """create teacher, student model and distiller"""
    # Model creation (for more output comment in summery and mflops)

    teacher = CNN32C3C3C5_BN_DA(input_shape, nb_classes, activation_dense=None)
    teacher._name = "Teacher_" + teacher._name
    #student = CNN32(input_shape, nb_classes, conv=(32, 32, 64), dense=256, use_dropout=True, activation_dense=None)
    student = Effnet(input_shape, nb_classes, activation_top=None)
    student._name = "Student_" + student._name


    # teacher model will later named model
    teacher.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam", metrics = ["accuracy"])
    
    student.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam", metrics = ["accuracy"])
    #student.summary()
    #print(f'Model {student._name} has MFLOPs :{get_flops(student)/1000000 }', )


    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller._name = student._name
    distiller.compile(
        optimizer=Adam(),
        metrics=[SparseCategoricalAccuracy()],
        student_loss_fn=SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=KLDivergence(),
        alpha=0.1,
        temperature=10)

    return distiller