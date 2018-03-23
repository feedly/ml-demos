import keras.applications.vgg16 as v

if __name__ == '__main__':
    model = v.VGG16()
    print(f'{model.count_params()} parameters??? Yowsers!')
