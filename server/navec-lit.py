from navec import Navec
path = './models/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)
print(navec['навек'])
model = navec.as_gensim
cosine_similar = model.most_similar('звери', topn=50)
print(cosine_similar)
print('---------------')
cosine_similar = model.most_similar('животное', topn=10)
print(cosine_similar)
print('---------------')
cosine_center = model.most_similar(positive=['лиса', 'волк', 'медведь', 'заяц'], topn=10)
print(cosine_center)
print('---------------')
cosine_center = model.most_similar(positive=['избушка', 'на','курьих', 'ножках', 'лес', 'молодец'], topn=10)
print(cosine_center)