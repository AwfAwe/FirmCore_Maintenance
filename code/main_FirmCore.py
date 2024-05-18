from FirmCore_Decompositions import FirmCore
from multilayer_graph import MultilayerGraph
from Information import Information
from time import time
import argparse


if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='FirmCore Decomposition and Maintenance in Multilayer Networks')

    # arguments
    parser.add_argument('d', help='dataset')
    parser.add_argument('o', help='operation')

    # options
    parser.add_argument('--save', dest='save', action='store_true', default=False ,help='save results')

    # read the arguments
    args = parser.parse_args()

    # dataset path
    dataset_path = "../datasets/" + args.d
    operation_path = "../datasets/" + args.o

    dataset_name = str(args.d)

    information = Information(args.d)
    information.print_dataset_name(dataset_path)

    # create the input graph and print its name
    start = int(round(time() * 1000))
    multilayer_graph = MultilayerGraph(dataset_path, operation_path)
    end = int(round(time() * 1000))
    print(" >>>> Preprocessing Time: ", (end - start)/1000.00, " (s)\n")

    # FirmCore algorithms
    FirmCore(multilayer_graph, information, dataset_name, save=args.save)
    









