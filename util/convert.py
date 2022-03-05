
def process_aspif_line(aspif:str, program_dict, literal_dict):
    asp = aspif.split(' ')
    if asp[0] == 'asp': # Version line
        return
    statement_type, statement = list(map(int, asp))
    if statement_type == 1:
        h = statement[0]
        m = statement[1]
        literals = []
        i = 0
        while statement:
            pass

