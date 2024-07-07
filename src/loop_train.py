from src.metricas import dice_coef, dice_loss
import numpy as np

def validar(modelo,epoca,val_generator):
    # Validación
    val_loss, val_dice = 0.0, 0.0
    num_val_batches = len(val_generator)
    val_dice_1 = 0.0

    val_dice_acum_1 = 0
    val_dice_acum_2 = 0
    val_loss_acum = 0
    
    for X_val, y_val in val_generator:
        
        val_loss, val_dice_1 = modelo.evaluate(X_val, y_val, verbose=0)
        y_pred_val = modelo.predict(X_val)

        val_dice = dice_coef(y_val.astype(np.float32), y_pred_val)

        val_dice_acum_1 = val_dice_acum_1 + val_dice_1
        val_dice_acum_2 = val_dice_acum_2 + val_dice
        val_loss_acum = val_loss_acum + val_loss


    val_loss_acum /= num_val_batches
    val_dice_acum_1 /= num_val_batches
    val_dice_acum_2 /= num_val_batches

    print("Val - Loss: {:.4f} Dice: {:.4f}".format(val_loss_acum, val_dice_acum_1))

    return val_loss_acum, val_dice_acum_1

def entrenar(modelo,epoca,train_generator):
    train_loss = 0.0
    train_dice = 0.0
    num_batches = len(train_generator)

    for batch_index, (X_batch, y_batch) in enumerate(train_generator):
        loss, dice = modelo.train_on_batch(X_batch, y_batch)
        train_loss += loss
        train_dice += dice

        print("Epoca:{} Batch:{}/{} Loss:{:.4f} Dice:{:.4f}".format(epoca, batch_index + 1, num_batches,loss, dice))
        break
    train_loss /= num_batches
    train_dice /= num_batches

    print("Train - Loss: {:.4f}, dice: {:.4f}".format(train_loss, train_dice))

    return train_loss, train_dice

"""
if __name__ == "__main__":
    
    epoca = 5
    best_loss = 1000000
    best_dice = 0
    for epoca in range(epocas):
        train_loss, train_dice = entrenar(modelo,epoca,train_generator)
        val_loss, val_dice = validar(modelo,epoca,val_generator)

        if val_loss < best_loss and val_dice > best_dice:
            best_loss = val_loss
            best_dice = val_dice
            #modelo.save("modelo_{epoca}_dice_{dice}_loss_{loss}.h5")
            modelo.save("modelo_{}_dice_{}_loss_{}.h5".format(epoca,loss, dice))
"""