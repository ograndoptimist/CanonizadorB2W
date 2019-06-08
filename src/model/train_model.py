from models import CanonizadorNetwork_1, CanonizadorNetwork_2
from keras import optimizers


if __name__ == '__main__':
    # Loading the necessary data to the model
    path_input = '../../data/processed/tensor_input_processed_by_length.csv'
    path_output = '../../data/processed/tensor_output_processed_by_length.csv'

    modelo_1 = CanonizadorNetwork_2()
    modelo_1.load_data(path_input, path_output)

    # Build the models' architecture
    modelo_1.build_model(embedding_dimension=30, lstm_dimension=30,
                         dense_units=256, optimizer=optimizers.Adam)

    # Train the model
    modelo_1.fit_model(epochs=2, batch_size=1024, shuffle=True, verbose=True)

    # Plott performances' model
    modelo_1.plot_model_performance('Training and validation loss.png', mode_1='loss',
                                    mode_2='val_loss', label_1='Training loss',
                                    label_2='Validation loss', xlabel='Epochs', ylabel='Loss',
                                    title='Training and validation loss',
                                    savefig='Training and validation loss.png')

    modelo_1.plot_model_performance('Training and validation accuracy.png',
                                    mode_1='acc', mode_2='val_acc', label_1='Training Accuracy',
                                    label_2='Validation accuracy', xlabel='Epochs', ylabel='Accuracy',
                                    title='Training and validation accuracy',
                                    savefig='Training and validation Accuracy.png')

    print(modelo_1.confusion_matrix())

    # Saving the weight's model
    # modelo_1.save()
