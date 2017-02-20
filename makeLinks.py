
import xlrd

W = xlrd.open_workbook('tes.xlsx')
p = W.sheet_by_index(1)
file = open("testfile.txt","w")
file1 = open("cmd.txt","r")
rows = []

file.write("model=BayesianModel([\n")
for i, row in enumerate(range(p.nrows)):
    r = []
    for j, col in enumerate(range(p.ncols)):
        r = str(p.cell_value(i, j))
        rows.append(r)
    if(rows[0] != rows[1]):
        if(i!=(p.nrows - 1)):
            file.write("('"+rows[0]+"',"+"'"+str(p.cell_value(i, j))+"'),\n")
        else:
            file.write("('"+rows[0]+"',"+"'"+str(p.cell_value(i, j))+"')\n")
    rows = []
file.write("])")
print("")
x= 0
x = eval("x+1")
print x


 

