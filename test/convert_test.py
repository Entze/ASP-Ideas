from util.convert import process_aspif_line

aspif_1 = """
asp 1 0 0
1 0 1 1 0 2 -2 1
1 0 1 2 0 2 -1 2
1 0 1 3 0 1 3
4 1 b 1 1
4 1 a 1 2
4 1 c 1 3
0
"""

aspif_2 = """
asp 1 0 0
5 1 2
1 0 1 2 0 1 -3
1 0 1 4 0 2 -2 1
1 0 1 3 0 2 -2 4
1 0 1 5 0 2 2 3
1 0 1 5 0 1 4
1 0 1 6 0 3 -5 -4 1
4 1 b 1 2
4 1 k 1 4
4 1 a 1 3
4 1 e 1 1
4 1 c 1 5
4 1 f 1 6
0
"""


def process_aspif_line_test_1():
    lines = aspif_1.split(sep='\n')
    program_dict = {}
    literal_dict = {}
    for line in lines:
        process_aspif_line(line, program_dict, literal_dict)
    print(program_dict)
    print(literal_dict)


def process_aspif_line_test_2():
    lines = aspif_2.split(sep='\n')
    program_dict = {}
    literal_dict = {}
    for line in lines:
        process_aspif_line(line, program_dict, literal_dict)
    print(program_dict)
    print(literal_dict)


process_aspif_line_test_1()
process_aspif_line_test_2()
