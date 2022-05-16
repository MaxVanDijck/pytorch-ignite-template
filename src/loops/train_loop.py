import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar

def create_engine(model, optimizer, criterion, config=None):

    # Define any training logic for iteration update
    def train_step(engine, batch):
        x, y = batch[0].to(idist.device()), batch[1].to(idist.device())

        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    # Define trainer engine
    engine = Engine(train_step)

    if idist.get_rank() == 0:
        # Add any custom handlers

        # Add progress bar showing batch loss value
        ProgressBar().attach(engine, output_transform=lambda x: {"batch loss": x})

    return engine

