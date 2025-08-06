0.99::eve.

action(reg_l2, overfitting) :- eve.
action(decr_lr, inc_loss) :- eve.
action(decr_lr, high_lr) :- eve.
action(inc_lr, low_lr) :- eve.

t(0.4)::action(inc_dropout, overfitting).
t(0.6)::action(data_augmentation, overfitting).

t(0.3)::action(decr_lr, underfitting).
t(0.5)::action(inc_neurons, underfitting).
t(0.45)::action(new_fc_layer).
t(0.45)::action(new_conv_layer).

t(0.85)::action(inc_batch_size, floating_loss).
t(0.15)::action(decr_lr, floating_loss).

