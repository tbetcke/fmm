! tools/fmm3d/driver.f90: the FMM3D side of the comparison of Phase 6 T9 (C6.8;
! docs/design/optimisation.md §8.1). Built by tools/fmm3d/build.sh against FMM3D's static
! library, outside the workspace; never linked into a crate.
!
!     fmm3d_driver PROBLEM PREFIX EPS[,EPS...] CALLS WARMUP VEC [FULL]
!
! PROBLEM is a problem file of `nd-fmm-bench --write-problem` (fmm-bench/src/problem.rs:
! the points and m charge vectors of a `scaling` workload, the sampled targets, the
! direct sum there). For every EPS in the list (FMM3D's requested precision; the tokens
! name the output files) the driver
!   1. times FMM3D's own setup once, apart from the evaluation: its subdivision rule
!      (`lndiv`), the tree (`pts_tree_mem`, `pts_tree_build`) and the sort of the points
!      into it (`pts_tree_sort`), called as `lfmm3d` calls them (FMM3D has no setup that
!      outlives a call: each call repeats it, so the call times below include it);
!   2. runs WARMUP untimed calls, then CALLS timed calls of the sources-only charge
!      interface with gradients, `lfmm3d_s_c_g`, one per charge vector 1 ... CALLS
!      (CALLS <= m); with CALLS = m it writes φ and ∇φ at the sampled targets to
!      PREFIX-eps<EPS>.out (fmm-bench/src/problem.rs, "The output file");
!      with FULL = 1 (and CALLS = m) also φ and ∇φ at every point to
!      PREFIX-eps<EPS>.full.out (the full output file of fmm-bench/src/problem.rs), for
!      the cross-check of the two codes' outputs (`nd-fmm-bench --compare`);
!   3. with VEC = 1, one timed call of the vectorised interface `lfmm3d_s_c_g_vec` with
!      nd = m densities, written to PREFIX-eps<EPS>.vec.out.
! It prints one `fmm3d:` line of key=value fields per EPS (the times in seconds by
! `omp_get_wtime`, the threads `omp_get_max_threads`); `nd-fmm-bench --check-output`
! measures the output files.
!
! The normalisation is checked first, on two unit charges at distance 1: CONVENTIONS §3.1
! takes φ = q/(4π r) and ∇φ = −q (x − y)/(4π r³). FMM3D up to v2.0.0 computed q/r
! (without the 1/(4π)), v2.1.0 computes q/(4π r) (its lfmm3d.f: "We use
! 1/(4\pi)*(1/r)"); the driver
! scales by what the check finds (1 or 1/(4π)) and stops on anything else.
!
! Helmholtz (CH1.4; docs/helmholtz/tasks/CH1.4.md), selected by the first argument; the
! Laplace usage above is unchanged:
!
!     fmm3d_driver helmholtz-check
!     fmm3d_driver helmholtz PROBLEM PREFIX ZK_RE ZK_IM TOKEN[,TOKEN...] [REPEAT]
!
! helmholtz-check is the two-point check of proposal §9.2 against the draft §3.15
! (docs/helmholtz/conventions-3.15-draft.md, item 1): charges q1 = 0.3 − 0.7i at
! x1 = (0,0,0) and q2 = 1 at x2 = (1,0,0); with E = e^{ik}/(4π) the draft has
! φ(x1) = q2 E, ∇φ(x1) = q2 (1 − ik) E (1,0,0), φ(x2) = q1 E, ∇φ(x2) = q1 (ik − 1) E (1,0,0)
! (φ = Σ q G_k, G_k = e^{ik r}/(4π r), the gradient in the target). It runs
! `hfmm3d_s_c_g` (eps 1e-12) and `h3ddirectcg` for k = 1, 2.5, 1 + 0.5i, 2.5 + i, 0.5i, 3i,
! prints one `helmholtz-check:` line of raw values per routine and k, and stops (status 1)
! unless every value is the draft's to 1e-12 relative; for the record only (outside the
! draft's branch Im k ≥ 0, Re k ≥ 0) also k = 1 − 0.5i and k = −1, marked `record`.
!
! helmholtz runs the check, then reads PROBLEM, a little-endian stream file:
!   'NDFMMHP1', int64 n, s, t, seed; real(8) sources(3,n); complex(16) charges(n) as
!   (re, im) pairs; int64 sampled(s), 0-based source indices; real(8) targets(3,t)
! and for every TOKEN, `direct` (FMM3D's `h3ddirectcg`, thresh 1e-300, so that only
! coincident points are dropped) or an eps (`hfmm3d_s_c_g`, or `hfmm3d_st_c_g` when
! t > 0; REPEAT calls, default 1, the fastest timed), writes PREFIX-<TOKEN>.out:
!   'NDFMMHO1', int64 s, t; complex(16) pot(s), grad(3,s) at the sampled sources (the self
!   term dropped), pottarg(t), gradtarg(3,t) at the targets
! and prints one `helmholtz:` line (seconds by `omp_get_wtime`, FMM3D's setup included).
! FMM3D's values are written as they come: the check shows they need no scale.
program fmm3d_driver
  use omp_lib
  implicit none
  character(len=4096) :: problem, prefix, list, token, arg
  character(len=8) :: magic
  integer(8) :: header(5), n, m, s, calls, warmup, vec, full, i, j, k, c, ier
  integer(8) :: ndiv, idivflag, nterms, nlevels, nboxes, ltree, iptr(8)
  integer(8) :: nlmin, nlmax, ifunif, iper, zero
  integer(8), allocatable :: idx(:), itree(:), isrc(:), isrcse(:,:)
  real(8), allocatable :: src(:,:), chg(:,:), q(:), pot(:), grad(:,:)
  real(8), allocatable :: potv(:,:), gradv(:,:,:), out(:,:,:), times(:), every(:,:,:)
  real(8), allocatable :: centers(:,:), boxsize(:)
  real(8) :: eps, scale, t0, t1, tree_s, vec_s, targ(3)
  real(8) :: two(3,2), qtwo(2), ptwo(2), gtwo(3,2)
  real(8), parameter :: pi = 3.14159265358979323846d0
  integer :: start, comma, unit

  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg)
    if (arg == 'helmholtz-check') then
      if (.not. helmholtz_check()) stop 1
      stop
    else if (arg == 'helmholtz') then
      call helmholtz_run()
      stop
    end if
  end if
  if (command_argument_count() /= 6 .and. command_argument_count() /= 7) then
    print '(a)', 'usage: fmm3d_driver PROBLEM PREFIX EPS[,EPS...] CALLS WARMUP VEC [FULL]'
    print '(a)', '       fmm3d_driver helmholtz-check'
    print '(a)', '       fmm3d_driver helmholtz PROBLEM PREFIX ZK_RE ZK_IM TOKEN[,TOKEN...] [REPEAT]'
    stop 2
  end if
  call get_command_argument(1, problem)
  call get_command_argument(2, prefix)
  call get_command_argument(3, list)
  call get_command_argument(4, arg)
  read (arg, *) calls
  call get_command_argument(5, arg)
  read (arg, *) warmup
  call get_command_argument(6, arg)
  read (arg, *) vec
  full = 0
  if (command_argument_count() == 7) then
    call get_command_argument(7, arg)
    read (arg, *) full
  end if

  ! The normalisation: unit charges at (0,0,0) and (1,0,0); at the first point
  ! φ = 1/(4π) and ∇φ = (1/(4π), 0, 0) in CONVENTIONS §3.1.
  two = 0
  two(1,2) = 1
  qtwo = 1
  call lfmm3d_s_c_g(1d-6, 2_8, two, qtwo, ptwo, gtwo, ier)
  scale = (1 / (4 * pi)) / ptwo(1)
  if (abs(scale - 1) > 1d-12 .and. abs(scale * 4 * pi - 1) > 1d-12) then
    print '(a,es23.15)', 'fmm3d: unknown normalisation: two-point potential ', ptwo(1)
    stop 1
  end if
  if (abs(gtwo(1,1) * scale * 4 * pi - 1) > 1d-12 .or. abs(gtwo(2,1)) > 1d-15 &
      .or. abs(gtwo(3,1)) > 1d-15) then
    print '(a,3es23.15)', 'fmm3d: unexpected two-point gradient ', gtwo(:,1)
    stop 1
  end if
  print '(a,es23.15,a,es23.15,a,es23.15)', 'fmm3d: normalisation two_point_pot=', ptwo(1), &
      ' two_point_grad_x=', gtwo(1,1), ' scale=', scale

  ! The problem file (little-endian, as every machine of the comparison).
  open (newunit=unit, file=problem, access='stream', form='unformatted', status='old', &
      action='read')
  read (unit) magic
  if (magic /= 'NDFMMPB1') then
    print '(a)', 'fmm3d: not a problem file: '//trim(problem)
    stop 1
  end if
  read (unit) header
  n = header(1)
  m = header(2)
  s = header(3)
  allocate (src(3,n), chg(m,n), idx(s))
  read (unit) src
  read (unit) chg
  read (unit) idx
  close (unit)
  if (calls < 0 .or. calls > m .or. warmup < 0) then
    print '(a)', 'fmm3d: CALLS must be in 0 ... m and WARMUP at least 0'
    stop 2
  end if
  print '(a,i0,a,i0,a,i0,a,i0,a,i0)', 'fmm3d: problem n=', n, ' vectors=', m, ' sampled=', &
      s, ' seed=', header(4), ' distribution=', header(5)

  allocate (q(n), pot(n), grad(3,n), out(4,m,s), times(max(calls, 1_8)))
  if (full == 1) allocate (every(4,m,n))
  start = 1
  do while (start <= len_trim(list))
    comma = index(list(start:), ',')
    if (comma == 0) then
      token = list(start:len_trim(list))
      start = len_trim(list) + 1
    else
      token = list(start:start + comma - 2)
      start = start + comma
    end if
    read (token, *) eps

    ! 1. FMM3D's setup, as lfmm3d does it for sources only with gradients.
    call l3dterms(eps, nterms)
    zero = 0
    nlmin = 0
    nlmax = 51
    ifunif = 0
    iper = 0
    t0 = omp_get_wtime()
    call lndiv(eps, n, zero, 1_8, 0_8, 2_8, 0_8, ndiv, idivflag)
    call pts_tree_mem(src, n, targ, zero, idivflag, ndiv, nlmin, nlmax, iper, ifunif, &
        nlevels, nboxes, ltree)
    allocate (itree(ltree), boxsize(0:nlevels), centers(3,nboxes))
    call pts_tree_build(src, n, targ, zero, idivflag, ndiv, nlmin, nlmax, iper, ifunif, &
        nlevels, nboxes, ltree, itree, iptr, centers, boxsize)
    allocate (isrcse(2,nboxes), isrc(n))
    call pts_tree_sort(n, src, itree, ltree, nboxes, nlevels, iptr, centers, isrc, isrcse)
    t1 = omp_get_wtime()
    tree_s = t1 - t0
    deallocate (itree, boxsize, centers, isrcse, isrc)

    ! 2. One density per call.
    do i = 1, warmup
      q = chg(1,:)
      call lfmm3d_s_c_g(eps, n, src, q, pot, grad, ier)
      call failed(ier)
    end do
    do k = 1, calls
      q = chg(k,:)
      t0 = omp_get_wtime()
      call lfmm3d_s_c_g(eps, n, src, q, pot, grad, ier)
      t1 = omp_get_wtime()
      call failed(ier)
      times(k) = t1 - t0
      do j = 1, s
        i = idx(j) + 1
        out(1,k,j) = pot(i) * scale
        out(2:4,k,j) = grad(:,i) * scale
      end do
      if (full == 1) then
        do i = 1, n
          every(1,k,i) = pot(i) * scale
          every(2:4,k,i) = grad(:,i) * scale
        end do
      end if
    end do
    if (calls == m) call write_output(trim(prefix)//'-eps'//trim(token)//'.out')
    if (calls == m .and. full == 1) call write_full(trim(prefix)//'-eps'//trim(token)//'.full.out')
    call sort(times(1:calls))

    ! 3. Every density in one call.
    vec_s = -1
    if (vec == 1) then
      allocate (potv(m,n), gradv(m,3,n))
      t0 = omp_get_wtime()
      call lfmm3d_s_c_g_vec(m, eps, n, src, chg, potv, gradv, ier)
      t1 = omp_get_wtime()
      call failed(ier)
      vec_s = t1 - t0
      do j = 1, s
        i = idx(j) + 1
        do k = 1, m
          out(1,k,j) = potv(k,i) * scale
          do c = 1, 3
            out(1 + c,k,j) = gradv(k,c,i) * scale
          end do
        end do
      end do
      deallocate (potv, gradv)
      call write_output(trim(prefix)//'-eps'//trim(token)//'.vec.out')
    end if

    write (*, '(a,a,a,i0,a,i0,a,i0,a,i0,a,i0,a,i0,a,f0.6,a,i0,a,i0)', advance='no') &
        'fmm3d: eps=', trim(token), ' n=', n, ' threads=', omp_get_max_threads(), &
        ' nterms=', nterms, ' ndiv=', ndiv, ' nlevels=', nlevels, ' nboxes=', nboxes, &
        ' tree_s=', tree_s, ' calls=', calls, ' warmup=', warmup
    if (calls > 0) then
      write (*, '(a,f0.6,a,f0.6,a,f0.6)', advance='no') ' call_min_s=', times(1), &
          ' call_median_s=', median(times(1:calls)), ' call_max_s=', times(calls)
    end if
    if (vec == 1) write (*, '(a,i0,a,f0.6)', advance='no') ' vec_nd=', m, ' vec_s=', vec_s
    if (calls == m) write (*, '(a,a)', advance='no') ' out=', &
        trim(prefix)//'-eps'//trim(token)//'.out'
    if (calls == m .and. full == 1) write (*, '(a,a)', advance='no') ' full_out=', &
        trim(prefix)//'-eps'//trim(token)//'.full.out'
    if (vec == 1) write (*, '(a,a)', advance='no') ' vec_out=', &
        trim(prefix)//'-eps'//trim(token)//'.vec.out'
    write (*, '(a)') ''
    flush (6)
  end do

contains

  subroutine failed(code)
    integer(8), intent(in) :: code
    if (code /= 0) then
      print '(a,i0)', 'fmm3d: the call failed, ier = ', code
      stop 1
    end if
  end subroutine failed

  subroutine write_output(path)
    character(len=*), intent(in) :: path
    integer :: u
    open (newunit=u, file=path, access='stream', form='unformatted', status='replace', &
        action='write')
    write (u) 'NDFMMOU1', m, s, out
    close (u)
  end subroutine write_output

  subroutine write_full(path)
    character(len=*), intent(in) :: path
    integer :: u
    open (newunit=u, file=path, access='stream', form='unformatted', status='replace', &
        action='write')
    write (u) 'NDFMMFU1', m, n, every
    close (u)
  end subroutine write_full

  ! Insertion sort, ascending: a few call times.
  subroutine sort(a)
    real(8), intent(inout) :: a(:)
    integer :: p, r
    real(8) :: v
    do p = 2, size(a)
      v = a(p)
      r = p - 1
      do while (r >= 1)
        if (a(r) <= v) exit
        a(r + 1) = a(r)
        r = r - 1
      end do
      a(r + 1) = v
    end do
  end subroutine sort

  real(8) function median(a)
    real(8), intent(in) :: a(:)
    integer :: h
    h = size(a)
    if (mod(h, 2) == 1) then
      median = a(h / 2 + 1)
    else
      median = (a(h / 2) + a(h / 2 + 1)) / 2
    end if
  end function median

  ! The two-point check of the header against the draft §3.15; true when every draft case
  ! matches.
  logical function helmholtz_check()
    complex(8) :: zks(8), zk, q(2), pot(2), grad(3,2), e, want(4), got(4)
    real(8) :: x(3,2), rel, side
    integer(8) :: ier
    integer :: c, r
    logical :: record, ok
    character(len=12) :: routine
    real(8), parameter :: pi = 3.14159265358979323846d0

    zks = [(1d0, 0d0), (2.5d0, 0d0), (1d0, 0.5d0), (2.5d0, 1d0), (0d0, 0.5d0), &
        (0d0, 3d0), (1d0, -0.5d0), (-1d0, 0d0)]
    x = 0
    x(1,2) = 1
    q = [(0.3d0, -0.7d0), (1d0, 0d0)]
    helmholtz_check = .true.
    do c = 1, size(zks)
      zk = zks(c)
      record = c > 6
      e = exp((0d0, 1d0) * zk) / (4 * pi)
      want = [q(2) * e, q(2) * (1 - (0d0, 1d0) * zk) * e, q(1) * e, &
          q(1) * ((0d0, 1d0) * zk - 1) * e]
      do r = 1, 2
        pot = 0
        grad = 0
        if (r == 1) then
          routine = 'hfmm3d_s_c_g'
          call hfmm3d_s_c_g(1d-12, zk, 2_8, x, q, pot, grad, ier)
          call failed(ier)
        else
          routine = 'h3ddirectcg'
          call h3ddirectcg(1_8, zk, x, q, 2_8, x, 2_8, pot, grad, 1d-300)
        end if
        got = [pot(1), grad(1,1), pot(2), grad(1,2)]
        rel = maxval(abs(got - want) / abs(want))
        side = maxval(abs(grad(2:3,:))) / abs(e)
        ok = rel <= 1d-12 .and. side <= 1d-15
        write (*, '(a,2es24.16,a,a)', advance='no') 'helmholtz-check: zk=', zk, &
            ' routine=', trim(routine)
        write (*, '(a,2es24.16,a,6es24.16)', advance='no') ' pot1=', pot(1), ' grad1=', &
            grad(:,1)
        write (*, '(a,2es24.16,a,6es24.16)', advance='no') ' pot2=', pot(2), ' grad2=', &
            grad(:,2)
        write (*, '(a,es10.3,a,es10.3)', advance='no') ' rel=', rel, ' side=', side
        if (record) then
          write (*, '(a,l1)') ' record draft=', ok
        else
          write (*, '(a,l1)') ' check draft=', ok
          if (.not. ok) helmholtz_check = .false.
        end if
      end do
    end do
    flush (6)
    if (.not. helmholtz_check) print '(a)', 'helmholtz-check: FMM3D does not compute the draft''s kernel'
  end function helmholtz_check

  ! The `helmholtz` mode of the header.
  subroutine helmholtz_run()
    character(len=4096) :: hproblem, hprefix, hlist, htoken, harg
    character(len=8) :: hmagic
    integer(8) :: hheader(4), hn, hs, ht, hrepeat, r, ier, jj
    integer(8), allocatable :: sampled(:)
    real(8), allocatable :: sources(:,:), targets(:,:), at(:,:)
    complex(8), allocatable :: charges(:), hpot(:), hgrad(:,:), spot(:), sgrad(:,:)
    complex(8), allocatable :: tpot(:), tgrad(:,:)
    complex(8) :: zk
    real(8) :: zre, zim, heps, best, ht0, ht1
    integer :: hstart, hcomma, hunit
    logical :: direct

    if (command_argument_count() /= 6 .and. command_argument_count() /= 7) then
      print '(a)', 'usage: fmm3d_driver helmholtz PROBLEM PREFIX ZK_RE ZK_IM TOKEN[,TOKEN...] [REPEAT]'
      stop 2
    end if
    call get_command_argument(2, hproblem)
    call get_command_argument(3, hprefix)
    call get_command_argument(4, harg)
    read (harg, *) zre
    call get_command_argument(5, harg)
    read (harg, *) zim
    call get_command_argument(6, hlist)
    hrepeat = 1
    if (command_argument_count() == 7) then
      call get_command_argument(7, harg)
      read (harg, *) hrepeat
    end if
    zk = cmplx(zre, zim, kind=8)
    if (.not. helmholtz_check()) stop 1

    open (newunit=hunit, file=hproblem, access='stream', form='unformatted', status='old', &
        action='read')
    read (hunit) hmagic
    if (hmagic /= 'NDFMMHP1') then
      print '(a)', 'helmholtz: not a Helmholtz problem file: '//trim(hproblem)
      stop 1
    end if
    read (hunit) hheader
    hn = hheader(1)
    hs = hheader(2)
    ht = hheader(3)
    allocate (sources(3,hn), charges(hn), sampled(hs), targets(3,max(ht, 1_8)))
    read (hunit) sources
    read (hunit) charges
    read (hunit) sampled
    if (ht > 0) read (hunit) targets(:,1:ht)
    close (hunit)
    if (minval(sampled) < 0 .or. maxval(sampled) >= hn) then
      print '(a)', 'helmholtz: a sampled index is outside the sources'
      stop 1
    end if
    allocate (at(3,hs), hpot(hn), hgrad(3,hn), spot(hs), sgrad(3,hs))
    allocate (tpot(max(ht, 1_8)), tgrad(3,max(ht, 1_8)))
    do jj = 1, hs
      at(:,jj) = sources(:,sampled(jj) + 1)
    end do
    print '(a,i0,a,i0,a,i0,a,i0,a,2es24.16)', 'helmholtz: problem n=', hn, ' sampled=', hs, &
        ' targets=', ht, ' seed=', hheader(4), ' zk=', zk

    hstart = 1
    do while (hstart <= len_trim(hlist))
      hcomma = index(hlist(hstart:), ',')
      if (hcomma == 0) then
        htoken = hlist(hstart:len_trim(hlist))
        hstart = len_trim(hlist) + 1
      else
        htoken = hlist(hstart:hstart + hcomma - 2)
        hstart = hstart + hcomma
      end if
      direct = htoken == 'direct'
      if (.not. direct) read (htoken, *) heps
      best = huge(1d0)
      do r = 1, hrepeat
        spot = 0
        sgrad = 0
        tpot = 0
        tgrad = 0
        ht0 = omp_get_wtime()
        if (direct) then
          call h3ddirectcg(1_8, zk, sources, charges, hn, at, hs, spot, sgrad, 1d-300)
          if (ht > 0) call h3ddirectcg(1_8, zk, sources, charges, hn, targets, ht, tpot, &
              tgrad, 1d-300)
        else if (ht > 0) then
          call hfmm3d_st_c_g(heps, zk, hn, sources, charges, hpot, hgrad, ht, targets, tpot, &
              tgrad, ier)
          call failed(ier)
        else
          call hfmm3d_s_c_g(heps, zk, hn, sources, charges, hpot, hgrad, ier)
          call failed(ier)
        end if
        ht1 = omp_get_wtime()
        best = min(best, ht1 - ht0)
      end do
      if (.not. direct) then
        do jj = 1, hs
          spot(jj) = hpot(sampled(jj) + 1)
          sgrad(:,jj) = hgrad(:,sampled(jj) + 1)
        end do
      end if
      open (newunit=hunit, file=trim(hprefix)//'-'//trim(htoken)//'.out', access='stream', &
          form='unformatted', status='replace', action='write')
      write (hunit) 'NDFMMHO1', hs, ht, spot, sgrad
      if (ht > 0) write (hunit) tpot(1:ht), tgrad(:,1:ht)
      close (hunit)
      write (*, '(a,a,a,i0,a,i0,a,es24.16,a,a)') 'helmholtz: token=', trim(htoken), &
          ' threads=', omp_get_max_threads(), ' repeat=', hrepeat, ' seconds=', best, &
          ' out=', trim(hprefix)//'-'//trim(htoken)//'.out'
      flush (6)
    end do
  end subroutine helmholtz_run

end program fmm3d_driver
