from nuclear_shape import NuclearShape 


shape_test = NuclearShape("example_real_data_0.cmm")

shape_test.ellipsoid_fit()
shape_test.print_metrics()

