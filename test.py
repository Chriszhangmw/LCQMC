from ddparser import DDParser
import networkx as nx
ddp = DDParser(use_pos=True)

print(ddp.parse("十三还是十四号"))
print(ddp.parse("十三，哦，十四号"))

# text1 = "教皇在 1980 年提出了一个方案，该方案被智利拒绝，在阿根廷获得了批准。"
# text2 = "教皇于 1980 年提出了一项被智利接受但被阿根廷驳回的解决方案。"
# if '被' in text1:
#     out = ddp.parse(text1)
#
#     length = len(out[0]['word'])
#
#     entity = ['n', 'f', 's', 'nz', 'nw', 'r', 'PER', 'LOC', 'ORG', 'TIME']
#     subject = []
#     object = []
#
#     edges = []
#     for idx in range(length):
#         num_head = out[0]['head'][idx]
#         if num_head != 0:
#             edges.append((idx, num_head - 1))
#
#         if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'SBV':
#             subject.append(idx)
#         if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'VOB':
#             object.append(idx)
#         if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'POB':
#             object.append(idx)
#
#     graph = nx.Graph(edges)
#
#     for sub in subject:
#         for obj in object:
#
#             set_sub = []
#             set_sub_ = []
#             set_obj = []
#             set_obj_ = []
#
#             for idx in range(len(out[0]['head'])):
#                 if out[0]['head'][idx] - 1 == sub:
#                     set_sub.append(idx)
#                 if idx == sub:
#                     set_sub.append(idx)
#             for i in range(set_sub[0], set_sub[-1] + 1):
#                 set_sub_.append(i)
#
#             for idx in range(len(out[0]['head'])):
#                 if out[0]['head'][idx] - 1 == obj:
#                     set_obj.append(idx)
#                 if idx == obj:
#                     set_obj.append(idx)
#             for i in range(set_obj[0], set_obj[-1] + 1):
#                 set_obj_.append(i)
#             sub_ = set_sub_[0]
#             obj_ = set_obj_[0]
#             number_path = nx.shortest_path(graph, source=sub, target=obj)
#             token_path = [out[0]['word'][idx] for idx in number_path]
#             token_output = out[0]['word']
#             token_output[number_path[-2]] = token_path[1]
#             token_output[number_path[1]] = ''
#
#             a = ''.join([out[0]['word'][idx] for idx in set_sub_])
#             b = ''.join([out[0]['word'][idx] for idx in set_obj_])
#
#             for i in set_sub_:
#                 token_output[i] = ''
#             for i in set_obj_:
#                 token_output[i] = ''
#
#             token_output[obj_] = a
#             token_output[sub_] = b
#             print(token_output)
#             if len(number_path) == 4 and '被' in token_path:
#                 print(text1)
#                 text1 = ''.join(token_output)
#                 print(text1 + '\n')
#
# if '被' in text2:
#     out = ddp.parse(text2)
#
#     length = len(out[0]['word'])
#
#     entity = ['n', 'f', 's', 'nz', 'nw', 'r', 'PER', 'LOC', 'ORG', 'TIME']
#     subject = []
#     object = []
#
#     edges = []
#     for idx in range(length):
#         num_head = out[0]['head'][idx]
#         if num_head != 0:
#             edges.append((idx, num_head - 1))
#
#         if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'SBV':
#             subject.append(idx)
#         if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'VOB':
#             object.append(idx)
#         if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'POB':
#             object.append(idx)
#
#     graph = nx.Graph(edges)
#
#     for sub in subject:
#         for obj in object:
#
#             set_sub = []
#             set_sub_ = []
#             set_obj = []
#             set_obj_ = []
#
#             for idx in range(len(out[0]['head'])):
#                 if out[0]['head'][idx] - 1 == sub:
#                     set_sub.append(idx)
#                 if idx == sub:
#                     set_sub.append(idx)
#             for i in range(set_sub[0], set_sub[-1] + 1):
#                 set_sub_.append(i)
#
#             for idx in range(len(out[0]['head'])):
#                 if out[0]['head'][idx] - 1 == obj:
#                     set_obj.append(idx)
#                 if idx == obj:
#                     set_obj.append(idx)
#             for i in range(set_obj[0], set_obj[-1] + 1):
#                 set_obj_.append(i)
#             sub_ = set_sub_[0]
#             obj_ = set_obj_[0]
#             number_path = nx.shortest_path(graph, source=sub, target=obj)
#             token_path = [out[0]['word'][idx] for idx in number_path]
#             token_output = out[0]['word']
#             token_output[number_path[-2]] = token_path[1]
#             token_output[number_path[1]] = ''
#
#             a = ''.join([out[0]['word'][idx] for idx in set_sub_])
#             b = ''.join([out[0]['word'][idx] for idx in set_obj_])
#
#             for i in set_sub_:
#                 token_output[i] = ''
#             for i in set_obj_:
#                 token_output[i] = ''
#
#             token_output[obj_] = a
#             token_output[sub_] = b
#             print(token_output)
#             if len(number_path) == 4 and '被' in token_path:
#                 print(text2)
#                 text2 = ''.join(token_output)
#                 print(text2 + '\n')