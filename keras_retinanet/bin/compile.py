import argparse
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


def compile_model(model_path, alpha=0.25, gamma=2.0, output_path="", lr=1e-5):
    print('Loading model, this may take a second...')
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    # compile model
    model.compile( 
            loss={  'regression'    : losses.smooth_l1(),        
                    'classification': losses.focal(alpha=alpha, gamma=gamma)},
                    optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)     
            )
    if output_path == "":
        output_path = model_path

    model.save(output_path)

def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--gamma', help='Gamma value for focal loss', type=float, default=2.0)
    parser.add_argument('--alpha', help='Alpha value for focal loss', type=float, default=0.25)
    parser.add_argument('--learning-rate', help='Initial learning rate for adam',  type=float, dest='lr', default=1e-5)
    parser.add_argument('--output-path', help='path for saving compiled model', default="")
    parser.add_argument('model_path', help='Model to compile')
    return parser.parse_args(args)

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    print("compiling model {} with gamma {} and alpha {}".format(args.model_path, args.alpha, args.gamma))
    compile_model(args.model_path, args.alpha, args.gamma, args.output_path, args.lr)

if __name__ == "__main__":
    main()
