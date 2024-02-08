from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import loralib as lora

def getLoraModel(model):

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"  # QUESTION_ANS
    )

    # prepare int-8 model for training
   # model = prepare_model_for_int8_training(model)


    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    lora.mark_only_lora_as_trainable(model)
    model.print_trainable_parameters()

    return model
