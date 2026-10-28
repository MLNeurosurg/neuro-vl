from transformers import Trainer


class NeuroVLTrainer(Trainer):
    def create_optimizer(self):
        """
        Create optimizer groups with a separate learning rate for the visual encoder.

        This follows the LLaVA trainer pattern: the visual encoder receives its own
        base LR, while the connector and language model use TrainingArguments.learning_rate.
        The HF Trainer scheduler is still shared, so all groups follow the same decay shape.
        """
        if self.optimizer is None:
            opt_model = self.model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            visual_encoder_no_decay = {
                f"visual_encoder.{name}"
                for name in opt_model.visual_encoder.no_weight_decay()
            }
            
            visual_encoder_lr = getattr(self.args, "visual_encoder_learning_rate", None)
            freeze_visual_encoder = getattr(self.args, "freeze_visual_encoder", False)
            use_visual_encoder_lr = (
                not freeze_visual_encoder
                and visual_encoder_lr is not None
                and visual_encoder_lr != self.args.learning_rate
            )

            if use_visual_encoder_lr:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            param
                            for name, param in opt_model.named_parameters()
                            if (
                                name in decay_parameters
                                and name.startswith("visual_encoder.")
                                and name not in visual_encoder_no_decay
                                and param.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": visual_encoder_lr,
                    },
                    {
                        "params": [
                            param
                            for name, param in opt_model.named_parameters()
                            if (
                                (
                                    name not in decay_parameters
                                    or name in visual_encoder_no_decay
                                )
                                and name.startswith("visual_encoder.")
                                and param.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": visual_encoder_lr,
                    },
                    {
                        "params": [
                            param
                            for name, param in opt_model.named_parameters()
                            if (
                                name in decay_parameters
                                and not name.startswith("visual_encoder.")
                                and param.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            param
                            for name, param in opt_model.named_parameters()
                            if (
                                name not in decay_parameters
                                and not name.startswith("visual_encoder.")
                                and param.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            param
                            for name, param in opt_model.named_parameters()
                            if name in decay_parameters and param.requires_grad
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            param
                            for name, param in opt_model.named_parameters()
                            if name not in decay_parameters and param.requires_grad
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Keep HF Trainer behavior for optimizers that provide their own params/model.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
