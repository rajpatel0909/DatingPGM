import xlrd

def getLinksOfNodes():
    file = xlrd.open_workbook('tes.xlsx')
    sheet = file.sheet_by_index(1)
    links = []
    
    for i, row in enumerate(range(sheet.nrows)):
        temp = [0,0]
        for j, col in enumerate(range(sheet.ncols)):
            temp[j] = sheet.cell_value(i,j)
        
        if temp[0] != temp[1]:
            links.append(tuple(temp))
        
    return links