import sys
import os
import keras
import keras_resnet

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

from .. import losses
from ..models.resnet import custom_objects


def compile_model(model_path):
    print('Loading model, this may take a second...')
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    # compile model
    model.compile( 
            loss={  'regression'    : losses.smooth_l1(),        
                    'classification': losses.focal()         },
                    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)     
            )
    model.save(model_path)

def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print("compiling model {}".format(arg))
        compile_model(arg)
            
if __name__ == "__main__":
    main()
