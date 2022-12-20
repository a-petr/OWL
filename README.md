## OWL:1.0
This is the first version of the emplementation of the method developed in 
"OWL: a rank aware regularization method for joint sparse recovery" by A.Petrosyan, K.Pieper and H.Tran

To install it as a package:

The files are structured in the following way:
|-- ./scripts                                   # auxelary code needed in the notebooks
|-- ./enviroment.yml                            # hypermarameters 
|-- ./data                                      # data for the notebooks
|-- ./requirements.txt                          # required libraries and modules
|-- ./src                                       # directory containing the main files
|   |-- ./src/data                              
|   |   `-- ./src/data/make_dataset.py          # main script generating data
|   |-- ./src/utils.py                          # auxelary methods needed in the main code
|   `-- ./src/basealgorithm.py                  # the class defining the algorithm
|-- ./notebooks                                 # notebooks for demonstration
|   `-- ./notebooks/figures                             

Example: 