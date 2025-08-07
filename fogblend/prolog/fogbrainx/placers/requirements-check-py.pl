hwTh(0.0).      bwTh(0.0).

sum_tuple_list([HW|HWs], (AccCPU, AccRAM, AccStorage)) :-
    HW = (CPU, RAM, Storage),
    sum_tuple_list(HWs, (AccCPU1, AccRAM1, AccStorage1)),
    AccCPU is AccCPU1 + CPU,
    AccRAM is AccRAM1 + RAM,
    AccStorage is AccStorage1 + Storage.
sum_tuple_list([],(0,0,0)).

nodeOk(S,N,P,AllocHW) :-
    service(S,SWReqs,HWReqs,IoTReqs),
    node(N,SWCaps,HWCaps,IoTCaps),
    swReqsOk(SWReqs,SWCaps),
    thingReqsOk(IoTReqs,IoTCaps),
    hwOk(N,HWCaps,HWReqs,P,AllocHW).

swReqsOk(SWReqs, SWCaps) :- subset(SWReqs, SWCaps).

thingReqsOk(TReqs, TCaps) :- subset(TReqs, TCaps).

hwOk(N,HWCaps,HWReqs,P,AllocHW) :-
    findall(HW,member((N,HW),AllocHW),HWs), 
    sum_tuple_list(HWs, CurrAllocHW), CurrAllocHW = (CurrAllocCPU, CurrAllocRAM, CurrAllocStorage),
    findall(HW, (member(on(S1,N),P), service(S1,_,HW,_)), OkHWs), 
    sum_tuple_list(OkHWs, NewAllocHW), NewAllocHW = (NewAllocCPU, NewAllocRAM, NewAllocStorage),
    hwTh(T), 
    HWCaps = (MaxCPU, MaxRAM, MaxStorage),
    HWReqs = (ReqCPU, ReqRAM, ReqStorage),
    MaxCPU >= ReqCPU + T - CurrAllocCPU + NewAllocCPU,
    MaxRAM >= ReqRAM + T - CurrAllocRAM + NewAllocRAM,
    MaxStorage >= ReqStorage + T - CurrAllocStorage + NewAllocStorage. 

linksOk(S,N,P,AllocBW) :-
    term_string(AllocBW, AllocBW_String),
    term_string([on(S,N)|P], P_String),
    py_call(utils_prolog:check_bw(AllocBW_String, P_String), Result),
    Result = 1.