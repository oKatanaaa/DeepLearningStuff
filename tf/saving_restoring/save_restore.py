import argparse
import tensorflow as tf

SAVE = 'save'
home_path = '/home/student401/study/tf/saving_restoring'



    
def save():
    var = tf.get_variable('var', [])
    var_assign_op = tf.assign(var, 1)
    
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        print('Var value is %s' % var.eval())
        sess.run(var_assign_op)
        save_path = saver.save(sess, home_path+'/model.ckpt')
        print('Var value is %s' % var.eval())
        print('Model saved to %s' % save_path)

        
def restore():
    var = tf.get_variable('var', [])
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, home_path+'/model.ckpt')
        print('Model restored')
        print('Var value is %s' % var.eval())
        
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing saving and restoring in TensorFlow script')
    parser.add_argument('action', type=str, help='action will be performed after running the script')

    args = parser.parse_args()
    if args.action == SAVE:
        save()
    else:
        restore()
        