import numpy as np
import pandas as pd
import re
min_len = 50
max_len = 100


def cun_all_line(file_path=""):
    # test_data = pd.read_csv("./data/TranData.csv")
    test_data = pd.read_csv(file_path)
    test_data = test_data.loc[:, ["id","text","unknownEntities"]]
    data = test_data.values.tolist()
    text_all_line = []
    id_all_line = []
    unknownEntities_all_line = []
    for line in data:
        id = line[0]
        unknownEntities = line[2]
        line_list = re.split(r'[,，。？?！!；;]', line[1])
        line = ""
        for i in range(len(line_list)):
            if len(line) + len(line_list[i]) > max_len:
                if(len(line)) > 0:
                    text_all_line.append(line)
                    id_all_line.append(id)
                    unknownEntities_all_line.append(unknownEntities)
                line = line_list[i]
                continue
            line += line_list[i]
            if len(line) > min_len:
                text_all_line.append(line)
                id_all_line.append(id)
                unknownEntities_all_line.append(unknownEntities)
                line = ""
                continue
        if len(line) > 0:
            text_all_line.append(line)
            id_all_line.append(id)
            unknownEntities_all_line.append(unknownEntities)
    df1 = pd.DataFrame({'id': id_all_line, 'text': text_all_line,'unknownEntities':unknownEntities_all_line})
    df1.to_csv('./Result.csv', header=True, index=False)
    return text_all_line,id_all_line,unknownEntities_all_line


def cun_all_line_test(file_path=""):
    # test_data = pd.read_csv("./data/Testdata.csv")
    test_data = pd.read_csv(file_path)
    test_data = test_data.loc[:, ["id","text"]]
    data = test_data.values.tolist()
    text_all_line = []
    id_all_line = []
    for line in data:
        id = line[0]
        line_list = re.split(r'[,，。？?！!；;]', line[1])
        line = ""
        for i in range(len(line_list)):
            if len(line) + len(line_list[i]) > max_len:
                if(len(line)) > 0:
                    text_all_line.append(line)
                    id_all_line.append(id)
                line = line_list[i]
                continue
            line += line_list[i]
            if len(line) > min_len:
                text_all_line.append(line)
                id_all_line.append(id)
                line = ""
                continue
        if len(line) > 0:
            text_all_line.append(line)
            id_all_line.append(id)
    df1 = pd.DataFrame({'id': id_all_line, 'text': text_all_line})
    df1.to_csv('./Result_test.csv', header=True, index=False,encoding="gb18030")
    return text_all_line,id_all_line


if __name__ == "__main__":
    all_line,id ,unknownEntities_all_line= cun_all_line("./data/TranData.csv")
    # print(len(all_line),len(id),len(unknownEntities_all_line))
    all_line, id = cun_all_line_test("./data/TestData.csv")
    # print(len(all_line), len(id))
    # test_data = pd.read_csv("./Result_test.csv")
    # test_data = test_data.loc[:, "text"]
    # data = test_data.values.tolist()
    # print(data[:10])
