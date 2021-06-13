from transformers import TFGPT2LMHeadModel 


def main(model):
    tf_model = TFGPT2LMHeadModel.from_pretrained(model, from_pt=True)
    tf_model.save_pretrained(model)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
