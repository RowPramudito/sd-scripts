try:
    from train_network import setup_parser, NetworkTrainer
    from library.train_util import read_config_from_file

    parser = setup_parser()
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
    print("\n\033[1m✅ Done! Go download your LoRA model.)")


except BaseException:
    import traceback
    import re

    tb = traceback.format_exc().split("\n")
    error_index = len(tb)

    for i, line in enumerate(tb):
        if re.match(r"^[A-Za-z-_]+Error:", line):
            error_index = i
            break

    tb_text = "\n".join(tb[:error_index])
    print(f"\n{tb_text}")

    if error_index < len(tb):
        tb_error = "\n".join(tb[error_index:])
        print(f"\nERROR:\n{tb_error}\n")