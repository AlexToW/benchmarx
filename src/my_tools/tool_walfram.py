

def get_python(s: str):
    res = ''
    for i in range(len(s) - 2):
        if s[i] == '^':
            res += '**'
        elif s[i].isdigit() and s[i+1] == ' ' and (s[i+2] == 'x' or s[i+2] == 'y'):
            res += f'{s[i]}*'
            i += 1
        elif (s[i] == 'x' or s[i] == 'y' or s[i].isdigit()) and s[i+1] == ' ' and s[i+2] == '(':
            res += (s[i] + '*')
            i += 1
        elif s[i] == ')' and s[i+1] == ' ' and s[i+2] == '(':
            res += (s[i] + '*')
            i += 1
        else:
            res += s[i]
    return res

def _main():
    '''
    (24 (8 x^3 - 4 x^2 (9 y + 4) + 6 x (9 y^2 + 8 y + 1) - 9 y (3 y^2 + 4 y + 1)) ((3 x^2 + 2 x (3 y - 7) + 3 y^2 - 14 y + 19) (x + y + 1)^2 + 1) + 12 (x^3 + x^2 (3 y - 2) + x (3 y^2 - 4 y - 1) + y^3 - 2 y^2 - y + 2) ((12 x^2 - 4 x (9 y + 8) + 3 (9 y^2 + 16 y + 6)) (2 x - 3 y)^2 + 30), 12 (x^3 + x^2 (3 y - 2) + x (3 y^2 - 4 y - 1) + y^3 - 2 y^2 - y + 2) ((12 x^2 - 4 x (9 y + 8) + 3 (9 y^2 + 16 y + 6)) (2 x - 3 y)^2 + 30) - 36 (8 x^3 - 4 x^2 (9 y + 4) + 6 x (9 y^2 + 8 y + 1) - 9 y (3 y^2 + 4 y + 1)) ((3 x^2 + 2 x (3 y - 7) + 3 y^2 - 14 y + 19) (x + y + 1)^2 + 1))

    '''
    s = '(24 (8 x^3 - 4 x^2 (9 y + 4) + 6 x (9 y^2 + 8 y + 1) - 9 y (3 y^2 + 4 y + 1)) ((3 x^2 + 2 x (3 y - 7) + 3 y^2 - 14 y + 19) (x + y + 1)^2 + 1) + 12 (x^3 + x^2 (3 y - 2) + x (3 y^2 - 4 y - 1) + y^3 - 2 y^2 - y + 2) ((12 x^2 - 4 x (9 y + 8) + 3 (9 y^2 + 16 y + 6)) (2 x - 3 y)^2 + 30), 12 (x^3 + x^2 (3 y - 2) + x (3 y^2 - 4 y - 1) + y^3 - 2 y^2 - y + 2) ((12 x^2 - 4 x (9 y + 8) + 3 (9 y^2 + 16 y + 6)) (2 x - 3 y)^2 + 30) - 36 (8 x^3 - 4 x^2 (9 y + 4) + 6 x (9 y^2 + 8 y + 1) - 9 y (3 y^2 + 4 y + 1)) ((3 x^2 + 2 x (3 y - 7) + 3 y^2 - 14 y + 19) (x + y + 1)^2 + 1))'
    with open('output.txt', 'w') as file:
        file.write(get_python(s))


if __name__ == '__main__':
    _main()