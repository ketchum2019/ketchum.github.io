# Spring #
## IOC

IOC也叫控制反转，将对象间的依赖关系交给Spring容器，使用配置文件来创建所依赖的对象，由主动创建对象改为了被动方式，实现解耦合。可以通过注解@Autowired和@Resource来注入对象，被注入的对象必须被下边的四个注解之一标注：

- @Controller
- @Service
- @Repository
- @Component

在Spring配置文件中配置 <context:annotation-config/>元素开启注解。还有一个概念是DI（依赖注入），和控制反转是同一个概念的不同角度的描述，即应用程序在运行时依赖IOC容器来动态注入对象需要的外部资源（对象等）。

## 什么是 Spring beans? ##

Spring beans 是那些形成 Spring 应用的主干的 java 对象。它们被 Spring IOC 容器初始化，装配，和管理。这些 beans 通过容器中配置的元数据创建。比如，以 XML 文件中<bean/> 的形式定义。 Spring 框架定义的 beans 都是单例 beans。


## 简单介绍一下 Spring bean 的生命周期 ##
bean 定义：在配置文件里面用<bean></bean>来进行定义。 
bean 初始化：有两种方式初始化:  
> 1. 在配置文件中通过指定 init-method 属性来完成 
2. 实现 org.springframwork.beans.factory.InitializingBean 接口 

bean 调用：有三种方式可以得到 bean 实例，并进行调用 
bean 销毁：销毁有两种方式 
> 1. 使用配置文件指定的 destroy-method 属性 
2. 实现 org.springframwork.bean.factory.DisposeableBean 接口 

## 请描述一下 Spring 的事务 ##
1. **声明式事务管理的定义**： 用在 Spring 配置文件中声明式的处理事务来代替代码式的处理事务。 这样的好处是， 事务管理不侵入开发的组件， 具体来说， 业务逻辑对象就不会意识到正在事务管理之中， 事实上也应该如此， 因为事务管理是属于系统层面的服务， 而不是业务逻辑的一部分， 如果想要改变事务管理策划的话， 也只需要在定义文件中重新配置即可，这样维护起来极其方便。
> 基于 `TransactionInterceptor` 的声明式事务管理： 两个次要的属性：  `transactionManager`， 用来指定一个事务治理器，并将具体事务相 关的操作请托给它；其他一个是 `Properties` 类型的`transactionAttributes` 属性，该属性的每一个键值对中，键指定的是方法名，方法名可以行使通配符，而值就是表现呼应方法的所运用的事务属性。

2. **编程式事物管理的定义**： 在代码中显式挪用 beginTransaction()、 commit()、 rollback()等事务治理相关的方法，
这就是编程式事务管理。Spring 对事物的编程式管理有基于底层 API 的编程式管理和基于 TransactionTemplate 的
编程式事务管理两种方式。

3. 编程式事务与声明式事务的区别： 
1）编程式事务是自己写事务处理的类，然后调用 
2）声明式事务是在配置文件中配置，一般搭配在框架里面使用！ 

## BeanFactory 常用的实现类有哪些 ##
Bean 工厂是工厂模式的一个实现，提供了控制反转功能，用来把应用的配置和依赖从正真的应用代码中分离。常用的 BeanFactory 实现有 `DefaultListableBeanFactory 、 XmlBeanFactory 、 ApplicationContext` 等。 `XMLBeanFactory`， 最常用的就是 `org.springframework.beans.factory.xml.XmlBeanFactory` ，它根据 XML 文件中的定义加载 beans。该容器从 XML 文件读取配置元数据并用它去创建一个完全配置的系统或应用。 

## Spring 支持的几种 bean 的作用域 ##
Spring 框架支持以下五种 bean 的作用域： 
1. **singleton** : bean 在每个 Spring ioc 容器中只有一个实例。 
2. **prototype**：一个 bean 的定义可以有多个实例。 
3. **request**：每次 http 请求都会创建一个 bean，该作用域仅在基于 web 的 Spring ApplicationContext 情形下有效。 
4. **session** ： 在 一 个 HTTP Session 中 ， 一 个bean定义对应一个实例 。该作用域仅在基于 web 的Spring ApplicationContext 情形下有效。 
5. **global-session**：在一个全局的 HTTP Session 中，一个 bean 定义对应一个实例。该作用域仅在基于 web 的Spring ApplicationContext 情形下有效。缺省的 Spring bean 的作用域是 Singleton。 

## 5 种不同方式的自动装配

Spring 装配包括手动装配和自动装配，手动装配是有基于 xml 装配、构造方法、setter 方法等 自动装配有五种自动装配的方式，可以用来指导 Spring 容器用自动装配方式来进行依赖注入。



## Spring 框架中的单例 bean 是线程安全的吗 ##
Spring 框架中的单例 bean 不是线程安全的

## 简单解释一下 spring 的 AOP ##
`AOP（Aspect Oriented Programming）`，即面向切面编程，可以说是OOP（Object Oriented Programming，面向对象编程）的补充和完善。OOP引入封装、继承、多态等概念来建立一种对象层次结构，用于模拟公共行为的一个集合。不过OOP允许开发者定义纵向的关系，但并不适合定义横向的关系，例如日志功能。日志代码往往横向地散布在所有对象层次中，而与它对应的对象的核心功能毫无关系对于其他类型的代码，如安全性、异常处理和透明的持续性也都是如此，这种散布在各处的无关的代码被称为横切（crosscutting），在OOP设计中，它导致了大量代码的重复，而不利于各个模块的重用。
**AOP技术恰恰相反，它利用一种称为"横切"的技术，剖解开封装的对象内部，并将那些影响了多个类的公共行为封装到一个可重用模块，并将其命名为"Aspect"，即切面。所谓"切面"，简单说就是那些与业务无关，却为业务模块所共同调用的逻辑或责任封装起来，便于减少系统的重复代码，降低模块之间的耦合度，并有利于未来的可操作性和可维护性**。使用"横切"技术，AOP把软件系统分为两个部分：核心关注点和横切关注点。业务处理的主要流程是核心关注点，与之关系不大的部分是横切关注点。横切关注点的一个特点是，他们经常发生在核心关注点的多处，而各处基本相似，比如权限认证、日志、事物。AOP的作用在于分离系统中的各种关注点，将核心关注点和横切关注点分离开来。AOP核心就是切面，它将多个类的通用行为封装成可重用的模块，该模块含有一组API提供横切功能。比如，一个日志模块可以被称作日志的AOP切面。根据需求的不同，一个应用程序可以有若干切面。在SpringAOP中，切面通过带有@Aspect注解的类实现。

## AOP 核心概念 

1. 切面（aspect）：类是对物体特征的抽象，切面就是对横切关注点的抽象 
2. 横切关注点：对哪些方法进行拦截，拦截后怎么处理，这些关注点称之为横切关注点。
3. 连接点（joinpoint）：被拦截到的点，因为 Spring 只支持方法类型的连接点，所以在 Spring中连接点指的就是被拦截到的方法，实际上连接点还可以是字段或者构造器。 
4. 切入点（pointcut）：对连接点进行拦截的定义 
5. 通知（advice）：所谓通知指的就是指拦截到连接点之后要执行的代码，通知分为前置、后置、异常、最终、环绕通知五类。 
6. 目标对象：代理的目标对象 
7. 织入（weave）：将切面应用到目标对象并导致代理对象创建的过程 
8. 引入（introduction）：在不修改代码的前提下，引入可以在运行期为类动态地添加一些方法
   或字段。

## AOP 两种代理方式

Spring 提供了两种方式来生成代理对象: JDKProxy 和 Cglib，具体使用哪种方式生成由AopProxyFactory 根据 AdvisedSupport 对象的配置来决定。默认的策略是如果目标类是接口，则使用 JDK 动态代理技术，否则使用 Cglib 来生成代理。 

**JDK动态接口代理**

> JDK 动态代理主要涉及到 java.lang.reflect 包中的两个类：Proxy 和 InvocationHandler。
> InvocationHandler 是一个接口，通过实现该接口定义横切逻辑，并通过反射机制调用目标类
> 的代码，动态将横切逻辑和业务逻辑编制在一起。Proxy 利用 InvocationHandler 动态创建
> 一个符合某一接口的实例，生成目标类的代理对象。 

**CGLib 动态代理**

> CGLib 全称为 Code Generation Library，是一个强大的高性能，高质量的代码生成类库，
> 可以在运行期扩展 Java 类与实现 Java 接口，CGLib 封装了 asm，可以再运行期动态生成新
> 的 class。和 JDK 动态代理相比较：JDK 创建代理有一个限制，就是只能为接口创建代理实例，
> 而对于没有通过接口定义业务方法的类，则可以通过 CGLib 创建动态代理。

# Spring源码解析

## Spring依赖注入的方式

1. 基于XML注入

> 通过构造方法注入
>
> 通过setter方法注入

2. 基于注解注入@Autowired

> 通过构造方法注入
>
> 通过setter方法注入
>
> 通过字段注入
>
> 通过方法参数注入

