###########################################
# Numerical Methods 2015
# Examination assignment
# Jakob Rørsted Mosumgaard
# Time-stamp: <2015-07-01 11:08:39 moss>
#
# Part A
###########################################

#
# Modules
#

# For printing to stderr
import sys

# For passing options
import argparse

# Import the different parts of the exercise
import partA


#
# General main-function, which handles the argument parsing
#
def main():
    # Initialize parser
    parser = argparse.ArgumentParser(description='Inverse iteration method')

    # Add and gather keywords
    parser.add_argument('part', help='Which part to run')
    parser.add_argument('--basic', action='store_true',
                        help='Basic test of the algorithm')
    parser.add_argument('--convergence', action='store_true',
                        help='Test of the convergence rate')
    args = parser.parse_args()

    # Run the correct main-function with the parsed arguments
    if vars(args)['part'].lower() == 'a':
        partA.mainA(**vars(args))
    else:
        print('You have to choose a part to run!', file=sys.stderr)

#
# If the file is called directly: Run the main!
#
if __name__ == '__main__':
    main()
