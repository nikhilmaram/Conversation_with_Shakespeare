# Deep Q&A

## Presentation

This work try to reproduce the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It use a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow.

To pretrain the dialog network:

    python main.py --train dialog

To pretrain the Shakespeare  network:

    python main.py --train shakespeare

To test the complete model
    python main.py --test --dialog_pretrained_model <dialog pretrained model> --shakespeare_pretrained_model <shakespeare pretrained model>
