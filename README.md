# Conversation with Shakespeare

## Presentation
This project aims to build a bot that will allow the user to converse in modern English and get the correspond- ing response in ”Shakespearean” or Elizabethan English. It uses sequence2sequence models to build this bot. Two models are employed, one after the other to the user’s input, to give an output in Elizabethan English. The first model(Dialog Model) generates the output to the user’s input in modern English, and the second model(Shakespeare Model) converts the generated output to Eliz- abethan English. Our results show that the bot is able to respond to modern English inputs with a reasonable quality of response. The outputs can only be evaluated qualitatively by human judgment as of now, due to the lack of an exist- ing modern English-”Shakespearean” English conversations dataset.

Dialog Model : This work try to reproduce the results in [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It use a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow.

Shakespeare Model : This part work tries to reproduce the results in [Shakespearizing Modern Language Using Copy-Enriched Sequence-to-Sequence Models](https://arxiv.org/abs/1707.01161). An Attention based pointer copy mechanism is used.


To pretrain the dialog network:

    python main.py --train dialog

To pretrain the Shakespeare  network:

    python main.py --train shakespeare

To test the complete model:

    python main.py --test --dialog_pretrained_model <dialog pretrained model> --shakespeare_pretrained_model <shakespeare pretrained model>
