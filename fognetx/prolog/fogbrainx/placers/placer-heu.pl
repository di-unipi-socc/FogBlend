placement(Services, P, Alloc, NewPlacement) :-
	once(preprocessing(Services, OrderedNodes, OrderedServices)), 
	once(early_checks(Services, P)),
	hplacement(OrderedServices, P, OrderedNodes, Alloc, NewPlacement).

hplacement([S|Ss], P, OrderedNodes, (AllocHW,AllocBW), Placement) :-
	member(N, OrderedNodes),
	nodeOk(S,N,P,AllocHW), linksOk(S,N,P,AllocBW),
	hplacement(Ss, [on(S,N)|P], OrderedNodes, (AllocHW,AllocBW), Placement).
hplacement([],P,_,_,P).

% Preprocessing to get ordered nodes and services
preprocessing(Services, OrderedNodes, OrderedServices) :-
	heuristicNodes(OrderedNodes),
	heuristicServices(Services, OrderedServices).

% Early checks before the placement
early_checks(Services, P) :-
	early_check_mapping(Services, P),
	early_check_sum(Services, P).

% Check if thre is at least one node that can host each service
early_check_mapping([S|Ss], P) :-
	service(S,_,_,_), node(N,_,_,_),
	nodeOk(S,N,P,[]), !,
	early_check_mapping(Ss, P).
early_check_mapping([], _).

% Check if the sum of resources required by the services does not exceed the resources of the nodes
early_check_sum(Services, P) :-
	findall(HWReq, (member(S, Services), service(S,_,HWReq,_)), ReqHWs),
	sum_tuple_list(ReqHWs, (ReqCPU, ReqRAM, ReqStorage)),
	findall(HWUsed, (member(on(S,_), P), service(S,_,HWUsed,_)), AllocHWs),
	sum_tuple_list(AllocHWs, (AllocCPU, AllocRAM, AllocStorage)),
	findall(HWCap, node(_,_,HWCap,_), AvailableHWs),
	sum_tuple_list(AvailableHWs, (MaxCPU, MaxRAM, MaxStorage)),
	MaxCPU >= ReqCPU + AllocCPU,
	MaxRAM >= ReqRAM + AllocRAM,
	MaxStorage >= ReqStorage + AllocStorage.
early_check_sum([], _).