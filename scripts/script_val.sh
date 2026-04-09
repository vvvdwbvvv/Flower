dataset=afhq_cat ## or celebahq or afhq_cat
model=ot  ## rectified for celebahq, gradient_step for method=pnp_gs (Hurault) or diffusion for method=pnp_diff (Zhu), ot otherwise.
eval_split=val
max_batch=8
batch_size_ip=4

### PNP FLOW
method=pnp_flow
for alpha in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.65 0.8 1.0
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp 1.0 alpha ${alpha} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps_pnp 100
done
done

## Flower
method=flower
for step in 50 100 150 200 300 500
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps ${step} device cuda:0
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps ${step} device cuda:0
done
done

## Flower_cov
method=flower_cov
for step in 50 100 150 200 300 500
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 1 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps ${step} device cuda:0
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} num_samples 5 max_batch ${max_batch} batch_size_ip ${batch_size_ip} steps ${step} device cuda:0
done
done

### OT ODE
method=ot_ode
for start_time in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4
do
for gamma in 'constant' 'gamma_t'
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} start_time ${start_time} max_batch ${max_batch} batch_size_ip ${batch_size_ip} gamma ${gamma}
done
done
done

### FLOW PRIORS (MAP)
method=flow_priors
batch_size_ip=1
for lmbda in 10 30 100 300 1000 3000 10000 30000 100000
do
for eta in 0.001 0.003 0.01 0.03 0.1
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} eta ${eta} lmbda ${lmbda} max_batch ${max_batch} batch_size_ip ${batch_size_ip}
done
done
done

## D FLow (Optim Source)
method=d_flow
for lmbda in 0.0003 0.001 0.003 0.01
do
for alpha in 0.05 0.1 0.2 0.3 0.5 0.7
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda ${lmbda} alpha ${alpha} max_batch ${max_batch} batch_size_ip ${batch_size_ip}
done
done
done

batch_size_ip=4


### PNP GRADIENT STEP
method=pnp_gs
model=gradient_step
for lr_pnp in 0.99 1.5 2.
do
for alpha in 0.1 0.3 0.5 0.8 1.0
do
for sigma_factor in 1. 1.1 1.2 1.4 1.5 1.6 1.8 2. 2.5 3. 4. 5. 6. 8. 10.
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lr_pnp ${lr_pnp} alpha ${alpha} max_batch ${max_batch} batch_size_ip ${batch_size_ip} algo pgd max_iter 100 sigma_factor ${sigma_factor}
done
done
done
done


### PNP DIFFUSION
method=pnp_diff
model=diffusion
for lmbda in 1.0 3.0 5.0 10.0 30.0 100.0 300.0 1000.0
do
for zeta in 0.1 0.2 0.3 0.5 0.7 1.0
do
for problem in motion_deblur
do
python main.py --opts dataset ${dataset} eval_split ${eval_split} model ${model} problem ${problem} method ${method} lmbda ${lmbda} zeta ${zeta} max_batch ${max_batch} batch_size_ip ${batch_size_ip}
done
done
done
