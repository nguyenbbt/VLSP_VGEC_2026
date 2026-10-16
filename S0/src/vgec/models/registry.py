def parameter_report(model, max_parameters=8_000_000_000, enforce=True):
    # PyTorch named_parameters removes duplicate Parameter objects for tied weights.
    parameters = list(model.parameters())
    total = sum(p.numel() for p in parameters)
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    result = dict(total_parameters=total, trainable_parameters=trainable,
                  counting_method="unique_unquantized_parameter_objects_including_classification_head",
                  within_limit=total <= max_parameters)
    if enforce and total > max_parameters:
        raise ValueError(f"Model has {total} parameters; limit is {max_parameters}")
    return result
