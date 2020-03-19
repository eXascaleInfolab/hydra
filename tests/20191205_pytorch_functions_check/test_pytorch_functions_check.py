#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import torch

def test_torch_max():
    """
    Description: test of the max function as it is used in our code
    Params:
        - No Params
    Returns:
        - No return value
    """
    print('### Max: used in our code to find the class that was predicted. In fact, for each image, two values are returned: [prob_class_0, prob_class_1].\n')

    print('1) Random tensor')
    print('>> tensor = torch.rand((5,2))')
    tensor = torch.rand((5,2))
    print(f'>> {tensor}')
    print()

    print('2) Compute torch.max on dimension 1 (row by row)')
    print('>> values, indices = torch.max(tensor, 1)')
    values, indices = torch.max(tensor, 1)
    print()

    print('3) Print returned values')
    print(f'>> values = {values}')
    print(f'>> indices = {indices}')
    print()
    print()


def test_torch_index_select():
    """
    Description: test of the torch_index_select function as it is used in our code
    Params:
        - No Params
    Returns:
        - No return value
    """
    print('### Index select: used in our code to get the exact prediction values for the positive class (class 1). In fact, for each image, two values are returned: [prob_class_0, prob_class_1].\n')

    print('1) Random tensor')
    print('>> tensor = torch.rand((5,2))')
    tensor = torch.rand((5,2))
    print(tensor)
    print()

    print('2) Compute torch.index_select for "tensor", on dimension 1 (=> indexing column by column => index 1 = element in the current row, column 1), torch.tensor([1]) = indices')
    print('>> values = torch.index_select(tensor, 1, torch.tensor([1]))')
    values = torch.index_select(tensor, 1, torch.tensor([1]))
    print()

    print('3) Print returned values')
    print(f'>> values = {values}')
    print()


def test_softmax():
    """
    Description: test of the function function as it is used in our code
    Params:
        - No Params
    Returns:
        - No return value
    """
    print('### Softmax: argument is the dimension which the softmax values are computed along')
    print('1) Random tensor')
    print('>> tensor = torch.rand((1,2))')
    tensor = torch.rand(32,2)
    print(tensor)
    print()

    print('2) Softmax function')
    print('>> value = torch.nn.softmax(tensor.data, dim=1)')
    value = torch.nn.functional.softmax(tensor.data, dim=1)
    print(value)
    print()


if __name__ == '__main__':
    test_torch_max()
    test_torch_index_select()
    test_softmax()
