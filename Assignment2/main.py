from LoadImages import img_generator
from GetModel import GetModel

if __name__ == '__main__':
    train_set, validation_set, test_set = img_generator()
    model = GetModel()
    # model.get_model_summary()
    # model.get_model()
    model.get_layer_info()
    model.set_trainable_layers(20)
    model.get_layer_info()
    # model.train_model(train_set, validation_set)
    # model.load_saved_model("./models/test_acc_0.8988.h5")
    # model.get_layer_info()
    # model.evaluate_model(test_set)
    # model.generate_full_report(test_set)
