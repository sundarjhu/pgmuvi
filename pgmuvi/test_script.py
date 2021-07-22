import numpy as np
import torch
import gpytorch
import pandas as pd
from gps import SpectralMixtureGPModel as SMG
import matplotlib.pyplot as plt



if __name__=="__main__":
    testdata = pd.read_csv("~/projects/betelgeuseScuba2/AlfOriAAVSO_Vband.csv")

    print(testdata)

    train_jd = torch.Tensor(testdata['JD'].to_numpy())
    train_mag = torch.Tensor(testdata['Magnitude'].to_numpy())

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = SMG(train_jd, train_mag, likelihood)
    print(model)
    print(model.covar_module)
    #help(model)

    for name, param in model.named_parameters():
        print(name, param)
    #[print(p) for p in model.parameters()]

    #exit()
    model.train()
    likelihood.train()

    training_iter = 100

    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_jd)
        loss = -mll(output, train_mag)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    print(model)

    for name, param in model.named_parameters():
        print(name, param)

    # Test points every 0.1 between 0 and 5
    test_x = torch.linspace(train_jd.min(), train_jd.max(), 1000)

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    # See https://arxiv.org/abs/1803.06058
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = likelihood(model(test_x))
        
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_jd.numpy(), train_mag.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()
