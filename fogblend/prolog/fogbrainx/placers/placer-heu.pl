placement(Services, P, Alloc, NewPlacement) :-
	once(preprocessing(Services, OrderedNodes, OrderedServices)), 
	once(early_checks(Services, P)),
	hplacement(OrderedServices, P, OrderedNodes, Alloc, NewPlacement).

% First clause: Try to place on already used nodes
hplacement([S|Ss], P, OrderedNodes, (AllocHW,AllocBW), Placement) :-
	alreadyUsedNode(N, P),
	nodeOk(S,N,P,AllocHW), linksOk(S,N,P,AllocBW),
	hplacement(Ss, [on(S,N)|P], OrderedNodes, (AllocHW,AllocBW), Placement).

% Second clause: Try unused nodes if first clause fails
hplacement([S|Ss], P, OrderedNodes, (AllocHW,AllocBW), Placement) :-
	member(N, OrderedNodes),
	\+ member(on(_, N), P),                % Only try unused nodes
	nodeOk(S,N,P,AllocHW), linksOk(S,N,P,AllocBW),
	hplacement(Ss, [on(S,N)|P], OrderedNodes, (AllocHW,AllocBW), Placement).

% Base case: No more services to place
hplacement([],P,_,_,P).

% Return one of the nodes already used in the placement
alreadyUsedNode(N, P) :-
	findall(N, distinct(N, member(on(_, N), P)), Ns), member(N, Ns).

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
	sum_tuple_list(ReqHWs, (ReqCPU, ReqGPU, ReqStorage)),
	findall(HWUsed, (member(on(S,_), P), service(S,_,HWUsed,_)), AllocHWs),
	sum_tuple_list(AllocHWs, (AllocCPU, AllocGPU, AllocStorage)),
	findall(HWCap, node(_,_,HWCap,_), AvailableHWs),
	sum_tuple_list(AvailableHWs, (MaxCPU, MaxGPU, MaxStorage)),
	MaxCPU >= ReqCPU + AllocCPU,
	MaxGPU >= ReqGPU + AllocGPU,
	MaxStorage >= ReqStorage + AllocStorage.
early_check_sum([], _).