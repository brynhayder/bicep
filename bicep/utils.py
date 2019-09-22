from types import SimpleNamespace

# Write docs, test, see if this needs more
class Registry:
    def __init__(self):
        self._register = dict()
    
    def __getitem__(self, key):
        return self._register[key]
    
    def __call__(self, name):
        def add(thing):
            self._register[name] = thing
            return thing
        return add


def evaluate(model, dataloader, metrics, device):
        outputs = list()
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            model_outputs = model(data)
            outputs.append(
                    tuple(
                        m(model, data, target, model_outputs) for m in metrics
                        )
                    )
        return outputs


def n_correct(logits, target):
        pred = logits.argmax(dim=1, keepdim=True)
        # Need to call .item() before the division to python casts to float
        # otherwise pytorch does integer division.
        return pred.eq(target.view_as(pred)).sum().item()
       

# def evaluate(model, dataloader, hooks, device):
        # for data, target in dataloader:
            # data = data.to(device)
            # target = target.to(device)
            # model_outputs = model(data)
            # state = SimpleNamespace(
                    # data=data,
                    # target=target,
                    # model=model,
                    # model_outputs=model_outputs
                # )
            # for hook in hooks:
                # hook(state)
