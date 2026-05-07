import torch
from tqdm import tqdm

class_num = 951

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def DDIM_inv_diffusion(x, model, b, eta, A_funcs, y, sigma_y, NFE, cls_fn=None, config=None, ntrials=1):
    with torch.no_grad():
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//NFE
        z0 = x.clone()

        # generate time schedule
        times = get_schedule_jump(NFE, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )

        alphabar_traj = torch.zeros(len(times),device=x.device)
        t_traj = torch.zeros(len(times),device=x.device,dtype=torch.long)
        for ii in range(len(alphabar_traj)):
            t_traj[ii] = max(times[ii]*skip,-1)
            alphabar_traj[ii] = compute_alpha(b, t_traj[ii].long())
        x0_avg = torch.zeros_like(x)
        for _ in range(ntrials):
            x = torch.randn_like(x)
            z0 = x.clone()
            xt_next, _ = DDIM_inv_fromxt2x1(x,z0,y,A_funcs,eta,sigma_y,alphabar_traj,t_traj,cls_fn,model,config)
            x0_avg = x0_avg + xt_next
        x0_avg = x0_avg / ntrials
    return [x0_avg], [xt_next]

def DDIM_inv_fromxt2x1(x,z0,y,A_funcs,eta_array,sigma_y,alphabar_traj,t_traj,cls_fn,model,config):
    def onestep_update(xt,et,at,at_next,eta,lambd_next,delta,zt):
        if 1-eta*(1-torch.exp(-2*delta)) >=0:
            coeff = (1-eta*(1-torch.exp(-2*delta)))**(1/2)
            eta_used = eta
        else:
            coeff = 0
            eta_used = 1/(1-torch.exp(-2*delta))
        coeff_x = ((1-at_next)/(1-at))**(1/2) * coeff
        coeff_x0 = (1-at_next)**(1/2) * torch.exp(lambd_next)*(1-torch.exp(-delta)*coeff)
        sigma_t = ((1-at_next)*eta_used*(1-torch.exp(-2*delta)))**(1/2)
        x0_pred = (xt - (1-at)**(1/2) * et) / (at**(1/2))
        xt_next = coeff_x * xt + coeff_x0*x0_pred + sigma_t*zt
        return xt_next, x0_pred
    
    n = x.size(0)
    Vtz0 = A_funcs.Vt_DDIM_inv(z0)
    Uty = A_funcs.Ut(y)
    for ii in range(len(t_traj)-1):
        i = t_traj[ii]
        
        # if j < i: # normal sampling 
        t = (torch.ones(n,device=i.device) * i).to(x.device)
        if ii == 0:
            xt = x.clone()
        else:
            xt = xt_next.clone()
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        

        at = alphabar_traj[ii]
        at_next = alphabar_traj[ii+1]
        if at_next >= 1:
            at_next = at + (1-at)/2

        # SDE Solver
        lambd = (1/2) * torch.log( at / (1 - at)) # (batchsize,1)    
        lambd_next = (1/2)*torch.log( at_next / (1 -at_next))
        delta =  lambd_next - lambd
        
        sgl_vals = A_funcs.singulars_DDIM_inv().squeeze()
        if sigma_y > 0:
            eta = eta_array[0]        
            eta0 = eta_array[1]
        else:
            eta0 = eta_array[0]
            
        z1 = torch.randn_like(xt)
        xt_next,x0_pred = onestep_update(xt,et,at,at_next,eta0,lambd_next,delta,z1)
        z2 = torch.randn_like(xt)
        xt_next_S,_ = onestep_update(xt,et,at,at_next,eta,lambd_next,delta,z2)
        
        mask_S = (sgl_vals>0).unsqueeze(0)
        temp = A_funcs.Vt_DDIM_inv(xt_next_S)
        xt_next_S = A_funcs.V_DDIM_inv(temp * mask_S)
        temp = A_funcs.Vt_DDIM_inv(xt_next)
        xt_next_Sc = xt_next.reshape(xt_next.shape[0],-1) - A_funcs.V_DDIM_inv(temp * mask_S)
        xt_next = (xt_next_Sc + xt_next_S).reshape(xt_next.shape[0],config.data.channels,config.data.image_size,config.data.image_size)

        def Add_cond(xt_next,y,A_funcs,at_next,sgl_vals,Uty,Vtz0,config):
            Vtxt_next = A_funcs.Vt_DDIM_inv(xt_next)
            VVtxt_next = A_funcs.V_DDIM_inv(Vtxt_next)
            VVtc_xt_next = xt_next.reshape(y.shape[0],-1) - VVtxt_next
            indx_s = torch.where((1-at_next).sqrt()*sgl_vals>at_next.sqrt()*sigma_y)[0]
            Vtxt_next[:,indx_s] = (at_next.sqrt())*Uty[:,indx_s]/sgl_vals[indx_s] + (1-at_next-at_next*sigma_y**2/sgl_vals[indx_s]**2).sqrt()*Vtz0[:,indx_s]
            VVtxt_next = A_funcs.V_DDIM_inv(Vtxt_next)
            xt_next = VVtxt_next + VVtc_xt_next
            xt_next = xt_next.reshape(y.shape[0],config.data.channels,config.data.image_size,config.data.image_size)
            return xt_next
        
        xt_next = Add_cond(xt_next,y,A_funcs,at_next,sgl_vals,Uty,Vtz0,config)
    return xt_next, x0_pred

# form RePaint
def get_schedule_jump(NFE, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, NFE - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = NFE
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, NFE)

    return ts

def _check_times(times, t_0, NFE):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= NFE, (t, NFE)
