#conding=utf-8

from core.config import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in config.get('gpus')])
import tensorflow as tf
from nets.factory import select_network
from core.learning_rate import multi_stepping, cosine_decay_with_warmup
from utils.preprocess import SingleSampleProcessor
from core.dataset import HierDataset
import shutil
import math
from utils.logger import Logger

class Task(object):
    def __init__(self):
        self.batches_per_epoch = {}
        self.class_num = {}
        self.data_names = [x['name'] for x in config.get('data')]
        # 构建模型
        model_config = config.get('model_config')
        network_name = model_config['name']
        del model_config['name']
        self.model = select_network(network_name)(**model_config)
        self.num_images_per_id = config.get('num_images_per_id')
        if self.num_images_per_id:
            self.batch_dataset = [(data['name'], data['num_id'] * self.num_images_per_id) for data in
                                  config.get('data')]
        else:
            self.batch_dataset = [(data['name'], data['batch_size']) for data in config.get('data')]

    def make_input(self):
        """"""
        get_input_fns = []
        self.class_nums = []
        for name, batch_size in self.batch_dataset:
            data = HierDataset(name, SingleSampleProcessor(), use_mask=False)
            data.make_dataset()
            self.batches_per_epoch[name] = int(math.ceil(data._sample_num / batch_size))
            self.class_num[name] = data.class_num
            self.class_nums.append(data.class_num)
            get_input_fn = data.base_batch_input(batch_size)
            get_input_fns.append(get_input_fn)

        file_names = []
        images = []
        labels = []
        for data_input in get_input_fns:
            file_names.append(data_input[0])
            images.append(data_input[1])
            labels.append(data_input[2])

        return file_names, images, labels

    def get_learning_rate(self, global_step):
        """"""
        step_per_epoch = max(self.batches_per_epoch.values())

        lr_config = config.get('learning_rate')
        lr_base = lr_config['lr_base']
        lr_policy = lr_config.get('policy', 'multi_step')
        print('task.batches_per_epoch: ', step_per_epoch, ' lr policy: ', lr_policy)
        warmup = lr_config.get('warmup', False)
        warmup_step = lr_config.get('warmup_step', None)
        warmup_epoch = lr_config.get('warmup_epoch', None)
        if warmup:
            assert (warmup_step or warmup_epoch)
            if warmup_step and warmup_epoch:
                raise ValueError('warmup_step and warmup_epoch only must be one')
            if not warmup_step and warmup_epoch:
                warmup_step = warmup_epoch * step_per_epoch
            print('warmup_step: ', warmup_step)

        if lr_policy == 'multi_step':
            boundaries_epoch = lr_config['boundaries']
            boundaries_step = [int(x * step_per_epoch) for x in boundaries_epoch]
            lr_decay = lr_config.get('lr_decay', 0.1)
            if warmup:
                boundaries_step = [warmup_step] + boundaries_step
                rates = [lr_base * (lr_decay ** i) for i in range(len(boundaries_step))]
                rates = [0.] + rates
            else:
                rates = [lr_base * (lr_decay ** i) for i in range(len(boundaries_step) + 1)]
            self.lr = multi_stepping(global_step, boundaries_step, rates, warmup=warmup)
        elif lr_policy == 'cosine':
            if not warmup:
                warmup_step = 0
            total_steps = config.get('total_epcohs') * step_per_epoch
            self.lr = cosine_decay_with_warmup(global_step,
                                               learning_rate_base=lr_base,
                                               total_steps=total_steps,
                                               warmup_learning_rate=0,
                                               warmup_steps=warmup_step,
                                               )
        return self.lr

    def get_log_tensors(self):
        return self.log_loss_tensors + [('lr', self.lr)]

    def make_model(self, images, labels, scope=None):
        losses, losses_names = self.model.loss(images, labels, self.class_nums, scope=scope)
        loss = tf.add_n(losses)
        regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regular_loss = tf.add_n(regular_losses, name='regular_loss')
        total_loss = tf.add(loss, regular_loss, name='total_loss')
        self.log_loss_tensors = list(zip(losses_names, losses)) + [('regular_loss', regular_loss)]

        return total_loss


    def pretrain_vars(self):
        excludes = ['global_step', 'Momentum']
        model_var = tf.global_variables()
        model_var = [v for v in model_var if all([e not in v.name for e in excludes])]
        return model_var

    def load_var(self, checkpoint_file, logger=None):
        """"""
        reader = tf.train.NewCheckpointReader(checkpoint_file)
        var_shape = reader.get_variable_to_shape_map()
        model_var = self.pretrain_vars()
        pretrained_var = {}
        for v in model_var:
            v_name = v.op.name
            if 'tower' in v_name:
                v_name = '/'.join(v_name.split('/')[1:])
            if v_name not in var_shape:
                if logger:
                    logger.write('%s not found' % v_name)
                else:
                    print('%s not found' % v_name)
            elif var_shape[v_name] != v.get_shape().as_list():
                # print(v_name, var_sahpe[v_name], v.get_shape().as_list())
                if logger:
                    logger.write('%s shape not match' % (v_name))
                else:
                    print('%s shape not match' % (v_name))
            else:
                pretrained_var[v_name] = v
        return pretrained_var

    def average_gradients(self, tower_grads):
        """"""
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self):
        if config.get('pretrain_task') and not config.get('pretrain_model_step') is None:
            checkpoint_file = os.path.join(config.get('root_path'),
                                           'pretrained/',
                                           config.get('pretrain_task'),
                                           'model.ckpt-%d' % config.get('pretrain_model_step')
                                           )
        else:
            checkpoint_file = None
        savedir = os.path.join(config.get('root_path'), 'train',  config.get('task_id'))
        logger = Logger(savedir)

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            file_names, images, labels = self.make_input()
            global_step = tf.train.get_or_create_global_step()
            lr = self.get_learning_rate(global_step)

            num_gpus = len(config.get('gpus'))
            batch_images = [[] for _ in range(num_gpus)]
            batch_labels = [{} for _ in range(num_gpus)]
            for idd, dataset in enumerate(images):
                dataset = tf.split(dataset, num_or_size_splits=num_gpus, axis=0)
                for i in range(num_gpus):
                    batch_images[i].append(dataset[i])

            for idd, dataset_label in enumerate(labels):
                for k in dataset_label:
                    dataset = tf.split(dataset_label[k], num_or_size_splits=num_gpus, axis=0)
                    for i in range(num_gpus):
                        if k not in batch_labels[i]:
                            batch_labels[i][k] = [dataset[i]]
                        else:
                            batch_labels[i][k].append(dataset[i])

            tower_grad = []
            opt = tf.train.MomentumOptimizer(lr if lr is not None else 0.0, momentum=0.9)

            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):  #
                for i in range(len(config.get('gpus'))):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope("tower_%d" % i):
                            # loss
                            loss = self.make_model(batch_images[i], batch_labels[i])
                            # 当前梯度
                            cur_grad = opt.compute_gradients(loss)
                            tower_grad.append(cur_grad)
                            # 变量共享
                            # tf.get_variable_scope().reuse_variables()
            grads = self.average_gradients(tower_grad)
            if config.get('multiply_var'):
                for i, (g, v) in enumerate(grads):
                    if g is not None:
                        op_name = v.op.name
                        for name in config.get('multiply_var'):
                            if name in op_name:
                                grads[i] = (g * config.get('multiply'), v)

            if config.get('clip_by_norm'):
                for i, (g, v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_norm(g, config.get('clip_by_norm')), v)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'tower_0')
            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(grads, global_step=global_step)

            if not os.path.exists(savedir) or tf.train.latest_checkpoint(savedir) is None:
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                if checkpoint_file:
                    logger.write('load checkpoint from: %s' % checkpoint_file)
                    pretrained_var = self.load_var(checkpoint_file, logger)
                    restorer = tf.train.Saver(pretrained_var)
            else:
                checkpoint_file = tf.train.latest_checkpoint(savedir)
                restorer = tf.train.Saver(tf.global_variables())
            shutil.copy(config.get('yaml'), os.path.join(savedir, 'config.yaml'))
            # fw = open(os.path.join(savedir, 'log.txt'), 'w')
            saver = tf.train.Saver(max_to_keep=config.get('save_max_to_keep'))
            log_tensors = dict(self.get_log_tensors())

            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

            import time
            t1 = time.time()
            with tf.Session(config=sess_config) as sess:
                init = tf.global_variables_initializer()
                _ = sess.run(init)
                if checkpoint_file:
                    restorer.restore(sess, checkpoint_file)
                i = sess.run(global_step)
                while i < config.get('total_epcohs') * max(self.batches_per_epoch.values()):
                    loss_np, _, log_tensors_np = sess.run([loss, train_op, log_tensors])
                    strs = ' '.join(["%s: %f" % (k, v) for k, v in log_tensors_np.items()])
                    epoch = i / max(self.batches_per_epoch.values()) + 1
                    step = i % max(self.batches_per_epoch.values()) + 1
                    if i % config.get('log_every_n_steps') == 0:
                        show_info = '[Epoch: %d][Step: %d] ' % (epoch, step) + strs
                        logger.write(show_info)
                    if i % config.get('save_every_n_steps') == 0 and i > 0:
                        show_info = 'save model at: %s/model.ckpt-%d' % (savedir, i)
                        logger.write(show_info)
                        saver.save(sess, os.path.join(savedir, 'model.ckpt'), global_step=global_step)
                    i += 1
                saver.save(sess, os.path.join(savedir, 'model.ckpt'), global_step=global_step)
                show_info = 'save model at: %s/model.ckpt-%d' % (savedir, i)
                logger.write(show_info)

            t2 = time.time()
            cost = t2 - t1
            cost_h = cost // 3600
            cost_m = (cost - cost_h * 3600) // 60
            cost_s = cost - cost_h * 3600 - cost_m * 60
            logger.write('cost time: %d H %d M %d s' % (cost_h, cost_m, cost_s))

if __name__ == '__main__':
    task = Task()
    task.train()
