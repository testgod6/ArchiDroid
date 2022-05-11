import json
import os
import re
from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory

DetectorFactory.seed = 0

num_0 = 0
num_1 = 0
num_2 = 0
num_3 = 0
mm = 0


def LoadJSON(file_path, file_name):
    app_activity = []
    activity_text = []
    activity_view = []
    activity_name = []
    start_node = []
    end_node = []
    node_dic = {}
    start_node_id = []
    end_node_id = []

    text_dic = {}
    view_dic = {}

    global num_0, num_1, num_2, num_3, mm

    ll = 0

    write_path = "./GetATG/"
    dir_name = write_path + os.path.splitext(file_name)[0]
    isExist = os.path.exists(dir_name)
    if not isExist:
        os.makedirs(dir_name)
    
    node_filename = dir_name +'/' + os.path.splitext(file_name)[0] + ".csv"
    edge_filename = dir_name +'/' + os.path.splitext(file_name)[0] + ".txt"

    # print(file_path+file_name)
    file = open(file_path + file_name, 'r', encoding="utf-8")
    # file = open(file_path + "br.com.sisponto.sispontomobile--10.json", 'r', encoding="utf-8")

    data = json.load(file)
    language = data["meta"]["defaultLanguage"]
    file.close()

    # nodes = []
    # activities = data['activities']

    transitions = data['transitions']
    transitions_nodes = []
    transitions_nodes.append(transitions)
    # print(transitions)

    idx_pointer = 0
    while len(transitions_nodes) > idx_pointer:
        transition = transitions_nodes[idx_pointer]
        if 'scr' in transition:
            transitions_nodes.extend((transition.pop('scr')))
        idx_pointer += 1

    a = transitions_nodes
    b = transition
    for i in range(len(transition)):
        act = transition[i]
        for k, v in act.items():
            if k == 'scr':
                if v != '':
                    start_activity_fullname = v
                    start_activity_name = v.split(".")[-1]
                    # print(start_activity_name)
                    start_node.append(start_activity_name)
                    if start_activity_name not in app_activity:
                        app_activity.append(start_activity_name)
                        activity_name.append(start_activity_fullname)
            if k == 'dest':
                if v != '':
                    end_activity_fullname = v
                    end_activity_name = v.split(".")[-1]
                    # print(end_activity_name)
                    end_node.append(end_activity_name)
                    if end_activity_name not in app_activity:
                        app_activity.append(end_activity_name)
                        activity_name.append(end_activity_fullname)

    m = app_activity
    # print(m)

    activities = data['activities']

    for i in range(len(activities)):
        act_name = activities[i]
        candidates = []
        for k, v in act_name.items():
            if k == 'name':
                if v in activity_name:
                    a_n = v.split(".")[-1]
                    # activities2.append(act_name)
                    act_text = []
                    view_list = []
                    for layout in act_name['layouts']:
                        node_inf = find_node_by_name(layout, 'textAttributes', 'children')
                        # print(layout)
                        if layout == '':
                            act_text.append("null")
                            # print(layout)
                        for textAttributes in node_inf:
                            # if 'value' in textAttributes['textAttributes'][0]:
                            a = "'value': "
                            # print(a)
                            if a in str(textAttributes['textAttributes']):
                                # print(textAttributes)
                                value = find_node_by_name(textAttributes, 'value', 'textAttributes')[0]['value']
                                if value != "":
                                    # print(value)
                                    act_text.append(value)
                                    # for a in textAttributes['textAttributes']:
                                #     m = a['value']
                                #     print(m)
                            else:
                                if textAttributes['textAttributes'][0] != "":
                                    # print(textAttributes['textAttributes'][0])
                                    act_text.append(textAttributes['textAttributes'][0])
                        view_inf = find_node_by_name(layout, 'viewClass', 'children')
                        for view in view_inf:
                            # print()
                            # print(view['viewClass'])
                            view_list.append(view['viewClass'])

                    if view_list:
                        nn = 1
                    else:
                        view_list.append("null")

                    if act_text:
                        # print(act_text)
                        nn = 1
                    else:
                        act_text.append("null")
                        # print(act_text)
                        # candidates.extend(node_inf)
                    # print(act_text)
                    text_dic[a_n] = ' '.join(act_text)
                    activity_text.append(act_text)

                    view_dic[a_n] = ' '.join(view_list)
                    activity_view.append(view_list)


    with open(node_filename, "a", encoding='utf-8') as wf:
        wf.write("id,label,text,view")
        wf.write("\n")

    null_t = 0
    null_v = 0
    for i in range(len(activity_name)):
        node_dic[activity_name[i].split(".")[-1]] = i
        # print(i)
        allact.append(activity_name[i].split(".")[-1])
        if activity_name[i].split(".")[-1] in text_dic.keys():
            text_file = text_dic[activity_name[i].split(".")[-1]]
        else:
            text_file = "null"
            # null_t = null_t + 1

        if activity_name[i].split(".")[-1] in view_dic.keys():
            view_file = view_dic[activity_name[i].split(".")[-1]]
        else:
            view_file = "null"
            # null_v = null_v+1

        text_file = text_file.replace('\n', '').replace('\r', '').replace(',', '')
        text_file = ' '.join(text_file.split())

        view_file = view_file.replace('\n', '').replace('\r', '').replace(',', '')
        view_file = ' '.join(view_file.split())

        if '\x00' in text_file:
            print(text_file)

        node_inf = str(i) + ',' + app_activity[i] + ',' + text_file + ',' + view_file
        # print(node_inf)

        # with open("./test.txt", "a", encoding='utf-8') as twf:
        #     twf.write(app_activity[i])
        #     twf.write("\t")
        #
        # with open("./doc.txt", "a", encoding='utf-8') as dwf:
        #     dwf.write(text_file)
        #     dwf.write("\n")
        #
        # with open("./view.txt", "a", encoding='utf-8') as dwf:
        #     dwf.write(view_file)
        #     dwf.write("\n")

        # with open(node_filename, "a", encoding='utf-8') as wf:
        #     wf.write(node_inf)
        #     wf.write("\n")
        #

    n = node_dic
    # print(n)
    # print(start_node)


    
    nodes.append(activities)
    idx_pointer = 0
    while len(nodes) > idx_pointer:
        node = nodes[idx_pointer]
        if 'name' in node:
            nodes.extend((node.pop('name')))
        idx_pointer += 1
    
    act = {}
    for i in range(len(node)):
        act = node[i]
        for k, v in act.items():
            if k == 'name':
                if v != '':
                    activity_name = v.split(".")[-1]
                    print(activity_name)
                    with open("./test.txt", "a", encoding='utf-8') as wf:
                        wf.write(activity_name)
                        wf.write("\t")
    
    with open("./test.txt", "a", encoding='utf-8') as wf:
        wf.write("\n")


def find_node_by_name(root, name, children_tag):
    candidates = []
    nodes = []

    nodes.append(root)
    while len(nodes) > 0:
        node = nodes.pop()
        if name in node:
            candidates.append(node)

        if children_tag not in node:
            continue

        for child in node[children_tag]:
            if child is None:
                continue
            nodes.append(child)

    return candidates


if __name__ == "__main__":
    name_count = 0
    file_path = "./new/"
    path_list = os.listdir(file_path)


    allact = []
    # print(path_list)
    for file_name in path_list:
        # print(file_name)
        LoadJSON(file_path, file_name)
