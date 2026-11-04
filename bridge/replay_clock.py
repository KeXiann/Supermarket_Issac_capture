"""Retrospective policy-release clock estimated from recorded LowState ticks.

LowState.tick is simulation milliseconds, not the host/controller clock.
The minimal submission does not retain exact Bridge command-application times;
interpolating tick/receipt anchors preserves measured slowdown, not exact DDS
latency. Original files are never rewritten by these helpers.
"""
import numpy as np


def recorded_control_clock(tick_ms, receive_ns, action_ns, source_time_s, physics_dt=.005):
    ticks=np.asarray(tick_ms,dtype=np.int64)
    receive=np.asarray(receive_ns,dtype=np.int64)
    action=np.asarray(action_ns,dtype=np.int64)
    source=np.asarray(source_time_s,dtype=float)
    if ticks.ndim!=1 or len(ticks)<2 or any(a.shape!=ticks.shape for a in (receive,action,source)):
        raise ValueError('clock arrays must be paired one-dimensional rows')
    if not np.isfinite(source).all() or np.any(ticks<0) or np.any(np.diff(source)<=0):
        raise ValueError('invalid source clock')
    delta=np.diff(ticks)
    if np.any((delta<0)&(delta>-(2**31))):raise ValueError('LowState tick moved backwards')
    unwrapped=ticks.astype(float)+np.r_[0,np.cumsum(delta<-(2**31))]*(2**32)
    # int(t*1000) in Bridge can truncate a 5ms boundary by 1ms.
    sim=np.rint(unwrapped/(physics_dt*1000))*physics_dt
    if np.any(np.diff(receive)<0) or np.any(np.diff(action)<=0):
        raise ValueError('host clocks moved backwards')
    keep=np.r_[True,np.diff(receive)>0]
    if keep.sum()<2:raise ValueError('insufficient LowState receipt anchors')
    if np.any((~keep)&(np.r_[0,np.diff(sim)]!=0)):
        raise ValueError('same LowState receipt has inconsistent simulation ticks')
    x=(receive[keep]-action[0])/1e9
    y=sim[keep]-sim[0]
    at=(action-action[0])/1e9
    mapped=np.interp(at,x,y)
    for outside,i,j in ((at<x[0],0,1),(at>x[-1],-2,-1)):
        mapped[outside]=y[i]+(at[outside]-x[i])*(y[j]-y[i])/(x[j]-x[i])
    if np.any(np.diff(mapped)<=0):raise ValueError('cannot infer a strictly increasing action clock')
    observation=sim-sim[0]-mapped[0]
    release=mapped-mapped[0]
    final_interval=max(physics_dt,float(np.median(np.diff(release)[-min(50,len(release)-1):])))
    duration=float(np.ceil((release[-1]+final_interval)/physics_dt-1e-9)*physics_dt)
    report={'method':'piecewise-linear LowState tick / receipt mapping at action-queue timestamps',
        'exact_bridge_application_times_retained':False,
        'source_span_s':float(source[-1]-source[0]),
        'physics_observation_span_s':float(sim[-1]-sim[0]),
        'physical_release_span_s':float(release[-1]),
        'measured_simulation_to_host_ratio':float((sim[-1]-sim[0])/(source[-1]-source[0])),
        'repeated_lowstate_rows':int(np.sum(np.diff(ticks)==0)),
        'physics_duration_s':duration,
        'note':'Scheduling is estimated to one physics-step precision; cannot restore missing initial state, DDS feedback delays or original encoder feedback.'}
    return release,observation,duration,report


def due_rows(times,previous,t):
    """Consume every control row, including multiple releases in one step."""
    end=int(np.searchsorted(times,t+1e-8,side='right'))
    return range(previous+1,end)
