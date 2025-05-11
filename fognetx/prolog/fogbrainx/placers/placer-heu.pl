placement(Services, P, Alloc, NewPlacement) :-
	once(preprocessing(Services,Alloc,CompatibleNodes)), 
	hplacement(Services, P, CompatibleNodes, Alloc, NewPlacement).

hplacement(Services, P, CompatibleNodes, Alloc, NewPlacement) :-
    candidateNodes(Services, P, CompatibleNodes, Alloc, CandidateNodes),
  	bestMatch(CandidateNodes,S,N), select(S,Services,NewServices),
    hplacement(NewServices, [on(S,N)|P], CompatibleNodes, Alloc, NewPlacement).
hplacement([], P, _, _, P).

bestMatch(CandidateNodes, S, N) :- sort(1, @=<, CandidateNodes, SortedNodes), member((_, S, Ns), SortedNodes), member(N, Ns). % sort by increasing candidates

candidateNodes([S|Services], P, Compatibles, (AllocHW, AllocBW), [(Len,S,Ps)|Ls]) :-
	findall(N, 
		( member((S,Ns),Compatibles), member((HWCaps,N,HWReqs),Ns), hwOk(N,HWCaps,HWReqs,P,AllocHW), linksOk(S,N,P,AllocBW) ) ,
		Ps),
    length(Ps,Len),
    candidateNodes(Services, P, Compatibles, (AllocHW, AllocBW), Ls).
candidateNodes([], _, _, _, []).

preprocessing(Services,(AllocHW, _),CompatibleNodes) :-
	compatibleNodes(Services,AllocHW,CompatibleNodes).

compatibleNodes([],_,[]).
compatibleNodes([S|Ss], AllocHW, [(S,Ns)|Cs]) :-
	compatible(S,AllocHW,Ns),
	compatibleNodes(Ss,AllocHW,Cs).

compatible(S,AllocHW,Ns) :-
	findall((HWCaps,N,HWReqs), quickNodeOK(S,AllocHW,N,HWCaps,HWReqs), Cs),
	% Annotate each item with the sum of connected link bw as a key
    maplist(add_bw_sum_key, Cs, Keyed),
    % Sort descending by key (negate sum to use keysort ascending)
    keysort(Keyed, SortedKeyed),
    % Remove keys and get the sorted list
    pairs_values(SortedKeyed, SortedCs),
    Ns = SortedCs.

add_bw_sum_key(( (CPU, RAM, Storage), N, HWReqs ), Sum-( ( (CPU, RAM, Storage), N, HWReqs ) )) :-
	% find all exit links from N
	findall(BW, link(N,_,_,BW), Exitlinks),
	% sum the bandwidth of the exit links
	sum_list(Exitlinks, TotBW),
    Sum is -TotBW. % negate for descending order


quickNodeOK(S,AllocHW,N,HWCaps,HWReqs) :-
	service(S, SWReqs, HWReqs, TReqs), 
	node(N, SWCaps, HWCaps, TCaps), 
	thingReqsOk(TReqs, TCaps), 
	swReqsOk(SWReqs, SWCaps),
	findall(HW,member((N,HW),AllocHW),HWs), sum_tuple_list(HWs, CurrAllocHW), CurrAllocHW = (CurrAllocCPU, CurrAllocRAM, CurrAllocStorage),
	hwTh(T),
	HWCaps = (MaxCPU, MaxRAM, MaxStorage),
    HWReqs = (ReqCPU, ReqRAM, ReqStorage),
    MaxCPU >= ReqCPU + T + CurrAllocCPU,
    MaxRAM >= ReqRAM + T + CurrAllocRAM,
    MaxStorage >= ReqStorage + T + CurrAllocStorage.